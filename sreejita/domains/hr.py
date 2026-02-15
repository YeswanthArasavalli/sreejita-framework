from __future__ import annotations

# =====================================================
# STANDARD & THIRD-PARTY IMPORTS
# =====================================================

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Governance: non-interactive backend
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from matplotlib.ticker import FuncFormatter

# =====================================================
# FRAMEWORK IMPORTS
# =====================================================

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# GOVERNED NUMERIC HELPERS (HR-SAFE)
# =====================================================

def safe_div(n: Any, d: Any) -> Optional[float]:
    """
    Governance-safe division.

    Guarantees:
    - No ZeroDivision
    - No NaN / Inf propagation
    - No exceptions
    - Returns None if unsafe
    """
    try:
        if n is None or d is None:
            return None
        if pd.isna(n) or pd.isna(d):
            return None
        if float(d) == 0.0:
            return None

        val = float(n) / float(d)

        if np.isnan(val) or np.isinf(val):
            return None

        return float(val)
    except Exception:
        return None


def safe_mean(series: Optional[pd.Series]) -> Optional[float]:
    """
    Governance-safe mean.

    Guarantees:
    - No coercion side-effects
    - Graceful degradation
    """
    if series is None or not isinstance(series, pd.Series):
        return None

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    return float(s.mean())


def safe_rate(numerator: Any, denominator: Any) -> Optional[float]:
    """
    Semantic alias for safe_div.
    Used for HR rates (attrition, absence, etc.)
    """
    return safe_div(numerator, denominator)


# =====================================================
# TIME & DATE HELPERS (HR-LIFECYCLE SAFE)
# =====================================================

HR_TIME_KEYWORDS: List[str] = [
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
    "date",
]


def _is_datetime_series(series: pd.Series) -> bool:
    """
    Conservative datetime verification.

    Guarantees:
    - No false positives on numeric IDs
    - Requires multiple valid parses
    """
    try:
        sample = series.dropna().iloc[:10]
        if sample.empty:
            return False

        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().sum() >= 3
    except Exception:
        return False


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Boundary-safe HR time column detector.

    Design guarantees:
    - Prefers employment lifecycle dates
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

    # Prefer lifecycle anchors over generic dates
    priority = ["hire", "joining", "start", "onboard"]

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

    Guarantees:
    - No mutation of original dataframe
    - Returns None if signal is weak
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

    Scope:
    - Workforce structure
    - Stability & retention
    - Capacity & people-related operational signals

    Governance:
    - Observational only
    - No optimization, targets, or thresholds
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

        df = df.copy(deep=False)

        # -----------------------------
        # TIME DETECTION (BOUNDARY-SAFE)
        # -----------------------------
        self.time_col: Optional[str] = detect_time_column(df)
        self.today: pd.Timestamp = pd.Timestamp.today()

        if self.time_col:
            self._time_series = coerce_datetime(df, self.time_col)
        else:
            self._time_series = None

        # -----------------------------
        # COLUMN RESOLUTION (LAZY, GUARDED)
        # -----------------------------
        def _res(col: Optional[str]) -> Optional[str]:
            return col if col and col in df.columns else None

        self.cols: Dict[str, Optional[str]] = {
            # Identity & Structure
            "employee_id": _res(
                resolve_column(df, "employee_id")
                or resolve_column(df, "employee")
            ),
            "department": _res(
                resolve_column(df, "department")
                or resolve_column(df, "team")
            ),

            # Compensation
            "salary": _res(
                resolve_column(df, "salary")
                or resolve_column(df, "compensation")
            ),

            # Demographics
            "gender": _res(
                resolve_column(df, "gender")
                or resolve_column(df, "sex")
            ),

            # Performance
            "performance_rating": _res(
                resolve_column(df, "rating")
                or resolve_column(df, "performance")
            ),

            # Employment Lifecycle
            "hire_date": _res(
                resolve_column(df, "hire_date")
                or resolve_column(df, "joining_date")
            ),
            "exit_date": _res(
                resolve_column(df, "exit_date")
                or resolve_column(df, "termination_date")
            ),
            "employment_status": _res(
                resolve_column(df, "status")
                or resolve_column(df, "active_status")
            ),

            # Attendance & Tenure
            "absence_days": _res(
                resolve_column(df, "absence")
                or resolve_column(df, "leave_days")
            ),
            "tenure": _res(
                resolve_column(df, "tenure")
                or resolve_column(df, "years_of_service")
            ),

            # Optional Financial Proxy (NON-OWNED)
            "revenue_proxy": _res(resolve_column(df, "revenue")),
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
        Execute the HR pipeline with stable, safe outputs.

        Guarantees:
        - Fixed output schema
        - Graceful degradation
        - No uncaught exceptions
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

            # -----------------------------
            # KPI COMPUTATION
            # -----------------------------
            kpis = self.calculate_kpis(df)
            if not isinstance(kpis, dict):
                kpis = {}

            if kpis and "_confidence" not in kpis:
                kpis["_confidence"] = {}

            self._last_kpis = kpis

            # -----------------------------
            # INSIGHTS & RECOMMENDATIONS
            # -----------------------------
            insights = self.generate_insights(df, kpis) or []
            recommendations = self.generate_recommendations(
                df, kpis, insights
            ) or []

            # -----------------------------
            # VISUALS (OPTIONAL)
            # -----------------------------
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

            # -----------------------------
            # FINAL ASSEMBLY
            # -----------------------------
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
        - No judgment or normative language
        - No sensitive or regulated inference
        - No dataframe mutation
        - Graceful degradation
        """
    
        kpis: Dict[str, Any] = {}
        confidence: Dict[str, float] = {}
        c = self.cols
    
        volume = len(df)
    
        # ==================================================
        # WORKFORCE SIZE & STRUCTURE
        # ==================================================
    
        if c.get("employee_id"):
            headcount = df[c["employee_id"]].nunique()
        else:
            headcount = volume
    
        kpis["headcount"] = headcount
        confidence["headcount"] = 0.85 if headcount else 0.6
    
        # ==================================================
        # ATTRITION SIGNALS (EVENT-BASED, NON-INFERENTIAL)
        # ==================================================
    
        if c.get("exit_date"):
            exit_series = pd.to_datetime(df[c["exit_date"]], errors="coerce")
            exits = exit_series.notna().sum()
    
            base = headcount if headcount else volume
    
            kpis["exit_event_count"] = exits
            kpis["exit_event_rate"] = safe_rate(exits, base)
    
            confidence["exit_event_count"] = 0.8
            confidence["exit_event_rate"] = 0.75
    
        elif c.get("employment_status"):
            status = df[c["employment_status"]].astype(str).str.lower()
            exit_events = status.str.contains("exit|left|resign|term", na=False)
    
            exits = int(exit_events.sum())
    
            kpis["exit_event_count"] = exits
            kpis["exit_event_rate"] = safe_rate(exits, volume)
    
            confidence["exit_event_count"] = 0.7
            confidence["exit_event_rate"] = 0.65
    
        # ==================================================
        # TENURE SIGNALS (DESCRIPTIVE)
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
    
            confidence["avg_tenure_days"] = 0.75
            confidence["median_tenure_days"] = 0.75
    
        elif c.get("tenure"):
            tenure = pd.to_numeric(df[c["tenure"]], errors="coerce").dropna()
    
            kpis["avg_reported_tenure"] = safe_mean(tenure)
            confidence["avg_reported_tenure"] = 0.7
    
        # ==================================================
        # COMPENSATION SIGNALS (DESCRIPTIVE ONLY)
        # ==================================================
    
        if c.get("salary"):
            salary = pd.to_numeric(
                df[c["salary"]]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce",
            ).dropna()
    
            kpis["avg_salary"] = safe_mean(salary)
            kpis["median_salary"] = (
                float(salary.median()) if not salary.empty else None
            )
    
            confidence["avg_salary"] = 0.7
            confidence["median_salary"] = 0.7
    
        # ==================================================
        # PRODUCTIVITY PROXIES (NON-OWNED, OBSERVATIONAL)
        # ==================================================
    
        if c.get("revenue_proxy") and c.get("employee_id"):
            total_revenue = pd.to_numeric(
                df[c["revenue_proxy"]], errors="coerce"
            ).sum()
    
            kpis["revenue_per_employee"] = safe_div(
                total_revenue, headcount
            )
    
            confidence["revenue_per_employee"] = 0.6
    
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
    
            confidence["avg_performance_rating"] = 0.65
            confidence["performance_rating_dispersion"] = 0.65
    
        # ==================================================
        # ABSENCE SIGNALS
        # ==================================================
    
        if c.get("absence_days"):
            absence = pd.to_numeric(
                df[c["absence_days"]], errors="coerce"
            ).dropna()
    
            kpis["avg_absence_days"] = safe_mean(absence)
            confidence["avg_absence_days"] = 0.65
    
        # ==================================================
        # KPI CONFIDENCE NORMALIZATION
        # ==================================================
    
        for k, v in confidence.items():
            if volume < 30:
                confidence[k] = max(0.45, v - 0.15)
    
        kpis["_confidence"] = confidence
    
        # ==================================================
        # KPI CAPABILITY MAP (EXECUTIVE SAFE)
        # ==================================================
    
        kpis["_kpi_capabilities"] = {
            "headcount": "workforce_size",
            "exit_event_count": "workforce_stability",
            "exit_event_rate": "workforce_stability",
            "avg_tenure_days": "workforce_experience",
            "median_tenure_days": "workforce_experience",
            "avg_reported_tenure": "workforce_experience",
            "avg_salary": "compensation",
            "median_salary": "compensation",
            "revenue_per_employee": "productivity_proxy",
            "avg_performance_rating": "performance_signal",
            "performance_rating_dispersion": "performance_signal",
            "avg_absence_days": "attendance",
        }
    
        self._last_kpis = kpis
        return kpis


    # ---------------- VISUALS (SMART SELECTION) ----------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        HR Visual Intelligence Engine
    
        Guarantees:
        - No thresholds or outcome-based importance
        - No sensitive or regulated inference
        - No dataframe mutation
        - Executive-safe language
        - Narrative diversity (not time-only, not histogram-only)
        - Visual hygiene (many → few)
        """
    
        output_dir.mkdir(parents=True, exist_ok=True)
        visuals: List[Dict[str, Any]] = []
        c = self.cols
    
        # --------------------------------------------------
        # SAFE SAVE WRAPPER (DEDUP-AWARE)
        # --------------------------------------------------
        seen_keys: Set[str] = set()
    
        def save(fig, name, caption, importance, category, axis):
            key = f"{category}:{axis}"
            if key in seen_keys:
                plt.close(fig)
                return
    
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
    
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "category": category,
                "axis": axis,
            })
    
            seen_keys.add(key)
    
        # ==================================================
        # WORKFORCE STRUCTURE (ENTITY / COMPOSITION)
        # ==================================================
    
        if c.get("department"):
            counts = df[c["department"]].value_counts()
            if counts.nunique() >= 3:
                fig, ax = plt.subplots(figsize=(7, 4))
                counts.head(8).plot(kind="bar", ax=ax)
                ax.set_title("Workforce Distribution by Organizational Unit")
                ax.set_ylabel("Employees")
    
                save(
                    fig,
                    "hr_workforce_by_unit.png",
                    "Distribution of workforce across organizational units",
                    0.90,
                    "workforce",
                    "entity",
                )
    
        # ==================================================
        # EXIT EVENT DISTRIBUTION (STRUCTURAL, NOT RATE)
        # ==================================================
    
        if c.get("department") and (c.get("exit_date") or c.get("employment_status")):
    
            if c.get("exit_date"):
                exit_mask = pd.to_datetime(
                    df[c["exit_date"]], errors="coerce"
                ).notna()
            else:
                status = df[c["employment_status"]].astype(str).str.lower()
                exit_mask = status.str.contains(
                    "exit|left|resign|term", na=False
                )
    
            exit_df = df.loc[exit_mask.fillna(False)]
    
            if not exit_df.empty:
                counts = exit_df[c["department"]].value_counts()
                if counts.nunique() >= 2:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    counts.head(6).plot(kind="bar", ax=ax)
                    ax.set_title("Recorded Exit Events by Unit")
                    ax.set_ylabel("Exit Events")
    
                    save(
                        fig,
                        "hr_exit_events_by_unit.png",
                        "Observed distribution of recorded exit events across units",
                        0.85,
                        "retention",
                        "entity",
                    )
    
        # ==================================================
        # TENURE DISTRIBUTION (EXPERIENCE / STABILITY)
        # ==================================================
    
        if c.get("hire_date"):
            hire = pd.to_datetime(df[c["hire_date"]], errors="coerce")
            tenure_days = (self.today - hire).dt.days.dropna()
    
            if tenure_days.nunique() >= 4:
                fig, ax = plt.subplots(figsize=(6, 4))
                tenure_days.plot(kind="hist", bins=10, ax=ax)
                ax.set_title("Tenure Distribution (Days)")
                ax.set_xlabel("Tenure (days)")
    
                save(
                    fig,
                    "hr_tenure_distribution.png",
                    "Observed distribution of workforce tenure",
                    0.80,
                    "stability",
                    "distribution",
                )
    
        # ==================================================
        # COMPENSATION DISTRIBUTION (DESCRIPTIVE ONLY)
        # ==================================================
    
        if c.get("salary"):
            salary = pd.to_numeric(
                df[c["salary"]]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce",
            ).dropna()
    
            if salary.nunique() >= 6:
                fig, ax = plt.subplots(figsize=(6, 4))
                salary.plot(kind="hist", bins=15, ax=ax)
                ax.set_title("Compensation Distribution")
                ax.set_xlabel("Compensation")
    
                save(
                    fig,
                    "hr_salary_distribution.png",
                    "Observed distribution of reported compensation values",
                    0.75,
                    "compensation",
                    "distribution",
                )
    
        # ==================================================
        # PERFORMANCE SIGNAL (DESCRIPTIVE, NON-RANKING)
        # ==================================================
    
        if c.get("performance_rating"):
            perf = pd.to_numeric(
                df[c["performance_rating"]], errors="coerce"
            ).dropna()
    
            if perf.nunique() >= 4:
                fig, ax = plt.subplots(figsize=(6, 4))
                perf.plot(kind="hist", bins=6, ax=ax)
                ax.set_title("Performance Rating Distribution")
                ax.set_xlabel("Rating")
    
                save(
                    fig,
                    "hr_performance_distribution.png",
                    "Distribution of recorded performance ratings",
                    0.70,
                    "performance",
                    "distribution",
                )
    
        # ==================================================
        # PRODUCTIVITY PROXY (STRUCTURAL, SINGLE METRIC)
        # ==================================================
    
        if c.get("revenue_proxy") and c.get("employee_id"):
            revenue = pd.to_numeric(
                df[c["revenue_proxy"]], errors="coerce"
            ).sum()
            hc = df[c["employee_id"]].nunique()
    
            value = safe_div(revenue, hc)
    
            if value is not None:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(["Revenue per Employee"], [value])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
                ax.set_title("Revenue per Employee (Proxy)")
    
                save(
                    fig,
                    "hr_revenue_per_employee.png",
                    "Observed revenue scaled by workforce size",
                    0.85,
                    "productivity",
                    "scalar",
                )
    
        # ==================================================
        # FINAL SELECTION — MANY → FEW (EXECUTIVE SAFE)
        # ==================================================
    
        # Enforce axis diversity (not all distributions)
        axes = {v["axis"] for v in visuals}
        if axes == {"distribution"} and len(visuals) > 1:
            visuals = visuals[:2]
    
        visuals.sort(
            key=lambda v: (
                -v["importance"],
                v["category"],
            )
        )
    
        return visuals[:4]


    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        HR / Workforce Composite Insight Engine
    
        Guarantees:
        - Composite-first logic
        - No thresholds, targets, or benchmarks
        - No judgment or alert language
        - No sensitive or regulated inference
        - Confidence-bounded insights
        - Graceful degradation
        """
    
        insights: List[Dict[str, Any]] = []
    
        # --------------------------------------------------
        # KPI EXTRACTION (SAFE)
        # --------------------------------------------------
        headcount = kpis.get("headcount")
        exit_rate = kpis.get("exit_event_rate")
        avg_tenure = kpis.get("avg_tenure_days")
        median_tenure = kpis.get("median_tenure_days")
        avg_absence = kpis.get("avg_absence_days")
        revenue_per_emp = kpis.get("revenue_per_employee")
        perf_dispersion = kpis.get("performance_rating_dispersion")
    
        # --------------------------------------------------
        # CONFIDENCE MODEL (HR-SAFE)
        # --------------------------------------------------
        def insight_conf(*signals: Any) -> float:
            present = sum(s is not None for s in signals)
            if present <= 1:
                return 0.55
            if present == 2:
                return 0.65
            if present == 3:
                return 0.75
            return 0.85
    
        # ==================================================
        # COMPOSITE INSIGHTS (PRIMARY — ≥7 WHEN POSSIBLE)
        # ==================================================
    
        # 1. Workforce Stability
        if exit_rate is not None and avg_tenure is not None:
            insights.append({
                "type": "composite",
                "theme": "stability",
                "title": "Workforce Stability Pattern",
                "so_what": (
                    "Observed exit activity together with tenure duration provides a "
                    "descriptive view of workforce stability and employee retention dynamics."
                ),
                "confidence": insight_conf(exit_rate, avg_tenure),
            })
    
        # 2. Experience Depth
        if avg_tenure is not None and median_tenure is not None:
            insights.append({
                "type": "composite",
                "theme": "experience",
                "title": "Experience Distribution Shape",
                "so_what": (
                    "Average and median tenure values together indicate how experience "
                    "is distributed between long-tenured and recently joined employees."
                ),
                "confidence": insight_conf(avg_tenure, median_tenure),
            })
    
        # 3. Workforce Capacity Utilization
        if avg_absence is not None and headcount is not None:
            insights.append({
                "type": "composite",
                "theme": "capacity",
                "title": "Workforce Capacity Utilization",
                "so_what": (
                    "Reported absence patterns viewed alongside workforce size "
                    "describe how available capacity fluctuates operationally."
                ),
                "confidence": insight_conf(avg_absence, headcount),
            })
    
        # 4. Productivity Context
        if revenue_per_emp is not None and headcount is not None:
            insights.append({
                "type": "composite",
                "theme": "productivity",
                "title": "Productivity Context",
                "so_what": (
                    "Revenue scaled by workforce size provides context for how "
                    "organizational output relates to staffing levels."
                ),
                "confidence": insight_conf(revenue_per_emp, headcount),
            })
    
        # 5. Performance Consistency
        if perf_dispersion is not None:
            insights.append({
                "type": "composite",
                "theme": "performance",
                "title": "Performance Rating Consistency",
                "so_what": (
                    "Variation in recorded performance ratings reflects how "
                    "evenly performance outcomes are distributed across employees."
                ),
                "confidence": insight_conf(perf_dispersion),
            })
    
        # 6. Organizational Scale Context
        if headcount is not None:
            insights.append({
                "type": "descriptive",
                "theme": "structure",
                "title": "Organizational Scale Visibility",
                "so_what": (
                    "Workforce size provides baseline context for interpreting "
                    "other people-related operational signals."
                ),
                "confidence": insight_conf(headcount),
            })
    
        # 7. Workforce Signal Coverage
        if len(kpis) >= 4:
            insights.append({
                "type": "descriptive",
                "theme": "data_quality",
                "title": "Workforce Signal Coverage",
                "so_what": (
                    "Available HR signals support multi-dimensional workforce analysis "
                    "across stability, capacity, and productivity dimensions."
                ),
                "confidence": insight_conf(*list(kpis.values())[:3]),
            })
    
        # ==================================================
        # FALLBACK (WEAK DATA SAFETY)
        # ==================================================
        if not insights:
            insights.append({
                "type": "atomic",
                "theme": "overview",
                "title": "Workforce Overview Available",
                "so_what": (
                    "Available data supports a descriptive overview of workforce structure "
                    "without strong composite signal confidence."
                ),
                "confidence": 0.5,
            })
    
        # ==================================================
        # FINAL GOVERNANCE (ORDER + CAP)
        # ==================================================
        insights.sort(
            key=lambda i: (
                i.get("type") != "composite",
                -i.get("confidence", 0),
            )
        )
    
        return insights[:6]


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
        - Advisory-only (no mandates, urgency, or priorities)
        - No thresholds, targets, or benchmarks
        - No sensitive or regulated actions
        - Insight-aware but not insight-dependent
        - Confidence-bounded
        - Graceful degradation
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        # --------------------------------------------------
        # SAFE KPI EXTRACTION
        # --------------------------------------------------
        exit_rate = kpis.get("exit_event_rate")
        avg_tenure = kpis.get("avg_tenure_days")
        median_tenure = kpis.get("median_tenure_days")
        avg_absence = kpis.get("avg_absence_days")
        revenue_per_emp = kpis.get("revenue_per_employee")
        perf_dispersion = kpis.get("performance_rating_dispersion")
        headcount = kpis.get("headcount")
    
        # --------------------------------------------------
        # CONFIDENCE BINDING (HR-SAFE)
        # --------------------------------------------------
        def rec_conf(*signals: Any) -> float:
            present = sum(s is not None for s in signals)
            if present <= 1:
                return 0.55
            if present == 2:
                return 0.65
            if present == 3:
                return 0.75
            return 0.85
    
        # ==================================================
        # WORKFORCE STABILITY (≥2)
        # ==================================================
        if exit_rate is not None or avg_tenure is not None:
            recommendations.extend([
                {
                    "theme": "workforce_stability",
                    "recommendation": (
                        "Consider periodically reviewing workforce stability patterns, "
                        "including how tenure distribution and recorded exit activity evolve over time."
                    ),
                    "confidence": rec_conf(exit_rate, avg_tenure),
                },
                {
                    "theme": "workforce_stability",
                    "recommendation": (
                        "Tenure signals may be useful when reflecting on institutional knowledge retention "
                        "and workforce continuity planning."
                    ),
                    "confidence": rec_conf(avg_tenure, median_tenure),
                },
            ])
    
        # ==================================================
        # CAPACITY & UTILIZATION (≥2)
        # ==================================================
        if avg_absence is not None or headcount is not None:
            recommendations.extend([
                {
                    "theme": "capacity_utilization",
                    "recommendation": (
                        "Observed absence patterns can provide contextual input when evaluating "
                        "workload balance and operational capacity across teams."
                    ),
                    "confidence": rec_conf(avg_absence, headcount),
                },
                {
                    "theme": "capacity_utilization",
                    "recommendation": (
                        "Workforce size and availability signals may support discussions "
                        "around resourcing flexibility and support mechanisms."
                    ),
                    "confidence": rec_conf(headcount),
                },
            ])
    
        # ==================================================
        # PRODUCTIVITY CONTEXT (≥2)
        # ==================================================
        if revenue_per_emp is not None:
            recommendations.extend([
                {
                    "theme": "productivity_context",
                    "recommendation": (
                        "Revenue scaled by workforce size can serve as contextual input "
                        "when reviewing staffing models or operational efficiency narratives."
                    ),
                    "confidence": rec_conf(revenue_per_emp),
                },
                {
                    "theme": "productivity_context",
                    "recommendation": (
                        "Productivity-related proxies may be useful alongside other "
                        "organizational signals when discussing role design or team structure."
                    ),
                    "confidence": rec_conf(revenue_per_emp, headcount),
                },
            ])
    
        # ==================================================
        # PERFORMANCE DISTRIBUTION (≥2)
        # ==================================================
        if perf_dispersion is not None:
            recommendations.extend([
                {
                    "theme": "performance_distribution",
                    "recommendation": (
                        "Variation in recorded performance ratings may be a useful reference "
                        "when reflecting on feedback consistency or role clarity."
                    ),
                    "confidence": rec_conf(perf_dispersion),
                },
                {
                    "theme": "performance_distribution",
                    "recommendation": (
                        "Performance distribution patterns can provide descriptive context "
                        "during talent development or review cycle discussions."
                    ),
                    "confidence": rec_conf(perf_dispersion, headcount),
                },
            ])
    
        # ==================================================
        # WORKFORCE ANALYTICS MATURITY (DESCRIPTIVE)
        # ==================================================
        if len(kpis) >= 4:
            recommendations.append({
                "theme": "analytics_maturity",
                "recommendation": (
                    "The availability of multiple workforce signals supports broader "
                    "people analytics discussions across stability, capacity, and productivity dimensions."
                ),
                "confidence": rec_conf(*list(kpis.values())[:3]),
            })
    
        # ==================================================
        # GRACEFUL FALLBACK
        # ==================================================
        if not recommendations:
            recommendations.append({
                "theme": "workforce_overview",
                "recommendation": (
                    "Current data supports a descriptive overview of workforce structure. "
                    "Additional signals may enable deeper workforce planning considerations."
                ),
                "confidence": 0.5,
            })
    
        # ==================================================
        # FINAL GOVERNANCE (ORDER + CAP)
        # ==================================================
        recommendations.sort(key=lambda r: -r.get("confidence", 0))
        return recommendations[:6]


# =====================================================
# DOMAIN DETECTOR (WITH CONFIDENCE FLOOR FIX)
# =====================================================
class HRDomainDetector(BaseDomainDetector):
    """
    Boundary-safe HR / Workforce domain detector

    Design guarantees:
    - Capability-based (not keyword-only)
    - Finance-safe (blocks P&L / ledger datasets)
    - Requires employee-centric semantics
    - Conservative confidence (no forced classification)
    """

    domain_name = "hr"

    # -------------------------------------------------
    # HR-EXCLUSIVE SEMANTIC ANCHORS
    # -------------------------------------------------
    CORE_TOKENS: Set[str] = {
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
        "performance",
    }

    # Contextual (supporting but insufficient alone)
    CONTEXT_TOKENS: Set[str] = {
        "department",
        "team",
        "role",
        "designation",
        "manager",
        "status",
    }

    # Explicit negative (ownership protection)
    FINANCE_TOKENS: Set[str] = {
        "revenue",
        "expense",
        "profit",
        "invoice",
        "ledger",
        "gl",
        "account",
        "cogs",
    }

    # -------------------------------------------------
    # DETECTION
    # -------------------------------------------------
    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        # ---------------- SAFETY ----------------
        if df is None or df.empty:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={},
            )

        cols = [str(c).lower() for c in df.columns]

        # ---------------- NEGATIVE GATE (FINANCE) ----------------
        finance_hits = [
            c for c in cols
            if any(t in c for t in self.FINANCE_TOKENS)
        ]

        # Strong finance signature → never HR
        if len(finance_hits) >= 3:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={"finance_hits": finance_hits},
            )

        # ---------------- HR SIGNALS ----------------
        core_hits = [
            c for c in cols
            if any(t in c for t in self.CORE_TOKENS)
        ]

        context_hits = [
            c for c in cols
            if any(t in c for t in self.CONTEXT_TOKENS)
        ]

        unique_core = set(core_hits)
        unique_context = set(context_hits)

        # ---------------- MINIMUM VIABILITY ----------------
        # HR must have employee-centric semantics
        if len(unique_core) < 2:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={
                    "core_hits": list(unique_core),
                    "context_hits": list(unique_context),
                },
            )

        # ---------------- CONFIDENCE (CAPABILITY-WEIGHTED) ----------------
        # Core signals dominate, context only boosts
        confidence = 0.35
        confidence += min(len(unique_core) * 0.15, 0.40)
        confidence += min(len(unique_context) * 0.05, 0.20)

        confidence = round(min(confidence, 0.95), 2)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals={
                "core_signal_count": len(unique_core),
                "context_signal_count": len(unique_context),
                "core_columns": sorted(unique_core),
                "context_columns": sorted(unique_context),
            },
        )


# =====================================================
# REGISTRATION (FRAMEWORK CONSISTENT)
# =====================================================
def register(registry):
    registry.register(
        "hr",
        HRDomain,
        HRDomainDetector,
    )

