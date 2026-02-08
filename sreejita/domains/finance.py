# =====================================================
# Finance Domain — Block 1
# Imports · Helpers · Time Detection
# =====================================================
from __future__ import annotations

# ===============================
# Standard Library Imports
# ===============================

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import warnings

# ===============================
# Third-Party Imports
# ===============================

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ===============================
# Framework Imports
# ===============================

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

warnings.filterwarnings("ignore")

# =====================================================
# GENERIC SAFE HELPERS (FRAMEWORK STANDARD)
# =====================================================

def safe_divide(n, d):
    """
    Division helper with strict zero / null protection.
    Always returns NaN where division is invalid.
    Preserves scalar vs series semantics.
    """
    if isinstance(n, pd.Series) or isinstance(d, pd.Series):
        return np.where((d == 0) | pd.isna(d), np.nan, n / d)
    if d in (0, None) or pd.isna(d):
        return np.nan
    return n / d


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Safely coerces selected columns to numeric.
    Non-parsable values become NaN.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =====================================================
# TIME DETECTION (FINANCE-SAFE, BOUNDARY-SAFE)
# =====================================================

@dataclass
class TimeContext:
    time_column: Optional[str]
    granularity: str        # yearly | quarterly | monthly | irregular | none
    is_ordered: bool
    coverage_periods: int


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detects a likely time column using conservative heuristics.
    Avoids false positives from generic identifiers.
    """
    candidates = {
        "date", "timestamp", "period",
        "month", "year", "quarter",
        "fiscal_date", "fiscal_period",
        "reporting_date"
    }

    for col in df.columns:
        lcol = str(col).lower()
        if any(tok == lcol or lcol.endswith(tok) for tok in candidates):
            try:
                pd.to_datetime(df[col].dropna().iloc[0])
                return col
            except Exception:
                pass

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    return None


def infer_time_granularity(series: pd.Series) -> str:
    """
    Infers time granularity conservatively.
    Finance data is often irregular; do not over-classify.
    """
    if series.isna().all():
        return "none"

    s = series.dropna().astype(str)

    if s.str.fullmatch(r"\d{4}").all():
        return "yearly"

    if s.str.contains("Q", case=False).any():
        return "quarterly"

    if s.nunique() >= 4:
        return "monthly"

    return "irregular"


def build_time_context(df: pd.DataFrame) -> TimeContext:
    """
    Builds a degradation-safe time context object.
    """
    time_col = detect_time_column(df)

    if not time_col:
        return TimeContext(None, "none", False, 0)

    series = df[time_col]

    try:
        parsed = pd.to_datetime(series, errors="coerce")
        is_ordered = parsed.is_monotonic_increasing
    except Exception:
        is_ordered = False

    return TimeContext(
        time_column=time_col,
        granularity=infer_time_granularity(series),
        is_ordered=is_ordered,
        coverage_periods=int(series.nunique(dropna=True))
    )


# =====================================================
# FINANCE ANALYTIC HELPERS (OPTIONAL SIGNALS)
# =====================================================

def benford_deviation(series: pd.Series) -> float:
    """
    Measures deviation from expected leading-digit distribution.
    This is a weak anomaly / integrity signal, not a fraud verdict.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return 0.0

    s = series.dropna().astype(str)
    digits = (
        s.str.lstrip("-")
         .str.replace(".", "", regex=False)
         .str[0]
    )

    digits = digits[digits.str.isnumeric()]

    if len(digits) < 100:
        return 0.0

    observed = digits.value_counts(normalize=True)
    expected = {str(d): np.log10(1 + 1 / d) for d in range(1, 10)}

    return float(
        sum(abs(observed.get(str(d), 0) - expected[str(d)]) for d in range(1, 10))
    )

def _has_numeric_column(df, names):
    names = {n.lower() for n in names}
    for col in df.columns:
        lcol = str(col).lower()
        if any(n in lcol for n in names) and pd.api.types.is_numeric_dtype(df[col]):
            return True
    return False

def _has_gl_account(df):
    for c in df.columns:
        l = str(c).lower()
        if ("gl" in l and ("account" in l or "code" in l)) or "ledger" in l:
            return True
    return False

# =====================================================
# VISUAL FORMATTERS (NO PLOTTING LOGIC)
# =====================================================

def human_currency_formatter(x, _):
    """
    Human-readable numeric formatter for finance visuals.
    """
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1e9:
        return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{x:.0f}"
# =====================================================
# Finance Domain — Block 2
# Domain Class · Preprocess
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Universal Finance Intelligence (Health, Efficiency, Risk)"

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses finance data with strict governance:
        - No data fabrication
        - No forced assumptions
        - Full signal availability tracking
        """

        df = df.copy()

        # -----------------------------
        # Time Detection (Governed)
        # -----------------------------
        self.time_context = build_time_context(df)
        self.time_col = self.time_context.time_column

        # -----------------------------
        # Signal Resolution (Soft, Safe)
        # -----------------------------
        def _resolve_any(candidates: List[str]) -> Optional[str]:
            for c in candidates:
                col = resolve_column(df, c)
                if col:
                    return col
            return None

        self.cols = {
            # Corporate P&L
            "revenue": _resolve_any(["revenue", "sales", "turnover"]),
            "expense": _resolve_any(["expense", "cost", "opex", "cogs"]),
            "profit": _resolve_any(["profit", "net_income", "ebit", "ebitda"]),

            # Balance Sheet
            "assets": _resolve_any(["assets", "total_assets"]),
            "equity": _resolve_any(["equity", "shareholder_equity"]),
            "debt": _resolve_any(["debt", "liabilities", "total_debt"]),

            # Banking / Credit (Optional)
            "receivables": _resolve_any(["accounts_receivable", "receivables"]),
            "loans": _resolve_any(["loan_amount", "loans"]),
            "npa": _resolve_any(["non_performing_assets", "npa"]),
            "collateral": _resolve_any(["collateral_value"]),
            "interest": _resolve_any(["interest_expense", "interest"]),

            # Market / Price (Optional)
            "close": _resolve_any(["close", "adj_close", "price"]),
            "volume": _resolve_any(["volume"]),
        }

        # -----------------------------
        # Signal Availability Registry
        # (Authoritative downstream contract)
        # -----------------------------
        self.available_signals = {
            k: v for k, v in self.cols.items() if v is not None
        }

        # -----------------------------
        # Numeric Safety (No Fabrication)
        # -----------------------------
        for col in self.available_signals.values():
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[,$]", "", regex=True)
                    .replace("", np.nan)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -----------------------------
        # Time Ordering (If Possible)
        # -----------------------------
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col).reset_index(drop=True)

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
        Execute the finance pipeline with stable, safe outputs.

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
                df, kpis
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

    # =====================================================
    # Finance Domain — Block 3
    # KPI Engine (Enterprise, Governed)
    # =====================================================
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes finance KPIs with strict governance:
        - No thresholds
        - No targets
        - No fabricated KPIs
        - Graceful degradation
        """
    
        kpis: Dict[str, Any] = {}
        c = self.available_signals
        time_col = self.time_col
    
        # =================================================
        # Sub-domain 1: Revenue Quality (≥7)
        # =================================================
        if "revenue" in c:
            s = df[c["revenue"]].dropna()
            if not s.empty:
                kpis.update({
                    "revenue_total": s.sum(),
                    "revenue_mean": s.mean(),
                    "revenue_median": s.median(),
                    "revenue_min": s.min(),
                    "revenue_max": s.max(),
                    "revenue_variability": s.std(),
                    "revenue_range": s.max() - s.min(),
                })
    
        # =================================================
        # Sub-domain 2: Cost Structure (≥7)
        # =================================================
        if "expense" in c:
            s = df[c["expense"]].dropna()
            if not s.empty:
                kpis.update({
                    "expense_total": s.sum(),
                    "expense_mean": s.mean(),
                    "expense_median": s.median(),
                    "expense_min": s.min(),
                    "expense_max": s.max(),
                    "expense_variability": s.std(),
                    "expense_range": s.max() - s.min(),
                })
    
        # =================================================
        # Sub-domain 3: Profitability (≥7)
        # =================================================
        if "profit" in c:
            s = df[c["profit"]].dropna()
            if not s.empty:
                kpis.update({
                    "profit_total": s.sum(),
                    "profit_mean": s.mean(),
                    "profit_median": s.median(),
                    "profit_min": s.min(),
                    "profit_max": s.max(),
                    "profit_variability": s.std(),
                    "profit_range": s.max() - s.min(),
                })
    
            if "revenue" in c:
                rev_mean = df[c["revenue"]].mean()
                prof_mean = df[c["profit"]].mean()
                if pd.notna(rev_mean):
                    kpis["profit_to_revenue_ratio"] = safe_divide(prof_mean, rev_mean)
    
        # =================================================
        # Sub-domain 4: Liquidity & Cash Proxies (≥7)
        # =================================================
        if "receivables" in c:
            s = df[c["receivables"]].dropna()
            if not s.empty:
                kpis.update({
                    "receivables_mean": s.mean(),
                    "receivables_median": s.median(),
                    "receivables_max": s.max(),
                    "receivables_variability": s.std(),
                })
    
        if "receivables" in c and "revenue" in c:
            kpis["receivables_to_revenue_ratio"] = safe_divide(
                df[c["receivables"]].mean(),
                df[c["revenue"]].mean()
            )
    
        if "assets" in c and "debt" in c:
            kpis["debt_to_assets_ratio"] = safe_divide(
                df[c["debt"]].mean(),
                df[c["assets"]].mean()
            )
    
        # =================================================
        # Sub-domain 5: Financial Stability & Risk (≥7)
        # =================================================
        if "revenue" in c:
            s = df[c["revenue"]].dropna()
            if len(s) > 2:
                kpis.update({
                    "revenue_stability_std": s.std(),
                    "revenue_stability_range": s.max() - s.min(),
                })
    
        if "expense" in c:
            s = df[c["expense"]].dropna()
            if len(s) > 2:
                kpis.update({
                    "expense_stability_std": s.std(),
                    "expense_stability_range": s.max() - s.min(),
                })
    
        if "profit" in c:
            s = df[c["profit"]].dropna()
            if len(s) > 2:
                kpis["profit_stability_std"] = s.std()
    
        # =================================================
        # Sub-domain 6: Banking Health (≥7, Optional)
        # =================================================
        if "loans" in c:
            s = df[c["loans"]].dropna()
            if not s.empty:
                kpis.update({
                    "loans_mean": s.mean(),
                    "loans_max": s.max(),
                    "loans_variability": s.std(),
                })
    
        if "npa" in c and "loans" in c:
            kpis["npa_to_loans_ratio"] = safe_divide(
                df[c["npa"]].mean(),
                df[c["loans"]].mean()
            )
    
        if "collateral" in c and "loans" in c:
            kpis["loan_to_collateral_ratio"] = safe_divide(
                df[c["loans"]].mean(),
                df[c["collateral"]].mean()
            )
    
        # =================================================
        # Sub-domain 7: Market Variability (≥7, Optional)
        # =================================================
        if "close" in c and time_col:
            s = (
                df[[time_col, c["close"]]]
                .dropna()
                .sort_values(time_col)[c["close"]]
            )
    
            if len(s) > 3:
                returns = s.pct_change().dropna()
                if not returns.empty:
                    kpis.update({
                        "price_return_mean": returns.mean(),
                        "price_return_std": returns.std(),
                        "price_return_min": returns.min(),
                        "price_return_max": returns.max(),
                        "price_level_mean": s.mean(),
                        "price_level_max": s.max(),
                        "price_level_min": s.min(),
                    })
    
        # =================================================
        # Sub-domain 8: Integrity / Anomaly Signals (≥7)
        # =================================================
        if "expense" in c:
            s = df[c["expense"]].dropna()
            if not s.empty:
                kpis.update({
                    "expense_benford_deviation": benford_deviation(s),
                    "expense_unique_ratio": s.nunique() / len(s),
                    "expense_zero_ratio": (s == 0).mean(),
                    "expense_negative_ratio": (s < 0).mean(),
                })
    
        return kpis

    # =====================================================
    # Finance Domain — Block 4
    # Visual Engine (Enterprise, Governed)
    # =====================================================
    
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Generates finance visuals with strict governance:
        - No thresholds
        - No targets
        - No annualization
        - Observational only
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.available_signals
        time_col = self.time_col
    
        # ---------------------------------
        # Save helper
        # ---------------------------------
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
    
        # ---------------------------------
        # Formatter
        # ---------------------------------
        def human_fmt(x, _):
            if pd.isna(x):
                return ""
            if abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
    
        # =================================================
        # MARKET VARIABILITY (≥3)
        # =================================================
        if "close" in c and time_col:
            s = (
                df[[time_col, c["close"]]]
                .dropna()
                .sort_values(time_col)
                .set_index(time_col)[c["close"]]
            )
    
            if len(s) > 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                s.plot(ax=ax)
                ax.set_title("Price Movement Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
                save(fig, "price_trend.png",
                     "Observed price movement over time", 0.92, "market")
    
                fig, ax = plt.subplots(figsize=(6, 4))
                s.plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Distribution of Price Levels")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "price_distribution.png",
                     "Observed distribution of price levels", 0.78, "market")
    
                returns = s.pct_change().dropna()
                if not returns.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    returns.plot(kind="hist", bins=20, ax=ax)
                    ax.set_title("Distribution of Price Returns")
                    save(fig, "return_distribution.png",
                         "Observed distribution of price changes", 0.76, "market")
    
        # =================================================
        # REVENUE & COST STRUCTURE (≥3)
        # =================================================
        if "revenue" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["revenue"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Revenue Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "revenue_distribution.png",
                 "Observed distribution of revenue values", 0.86, "corporate")
    
        if "expense" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["expense"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Expense Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "expense_distribution.png",
                 "Observed distribution of expense values", 0.84, "corporate")
    
        if "revenue" in c and "expense" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Revenue", "Expense"],
                [df[c["revenue"]].sum(), df[c["expense"]].sum()]
            )
            ax.set_title("Revenue and Expense Magnitudes")
            ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
            save(fig, "revenue_expense.png",
                 "Relative magnitude of revenue and expenses", 0.9, "corporate")
    
        # =================================================
        # PROFITABILITY (≥2)
        # =================================================
        if "profit" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["profit"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Profit Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "profit_distribution.png",
                 "Observed distribution of profit values", 0.82, "corporate")
    
        # =================================================
        # CAPITAL STRUCTURE (≥2)
        # =================================================
        if "debt" in c and "equity" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Debt", "Equity"],
                [df[c["debt"]].mean(), df[c["equity"]].mean()]
            )
            ax.set_title("Average Capital Components")
            ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
            save(fig, "capital_structure.png",
                 "Observed capital structure components", 0.83, "risk")
    
        # =================================================
        # BANKING HEALTH (≥2, OPTIONAL)
        # =================================================
        if "loans" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["loans"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Loan Value Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "loan_distribution.png",
                 "Observed distribution of loan values", 0.8, "banking")
    
        if "loans" in c and "npa" in c:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Loans", "Non-Performing Assets"],
                [df[c["loans"]].mean(), df[c["npa"]].mean()]
            )
            ax.set_title("Loan and Non-Performing Asset Levels")
            ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
            save(fig, "loan_npa_levels.png",
                 "Observed loan and non-performing asset levels", 0.79, "banking")
    
        # =================================================
        # INTEGRITY / ANOMALY (≥1)
        # =================================================
        if "expense" in c:
            dev = benford_deviation(df[c["expense"]])
            if dev > 0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(["Deviation"], [dev])
                ax.set_title("Expense Digit Distribution Deviation")
                save(fig, "expense_benford.png",
                     "Observed deviation in leading digit distribution", 0.75, "integrity")
    
        # =================================================
        # MANY → FEW (REPORT TRIM)
        # =================================================
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:6]

    # =====================================================
    # Finance Domain — Block 5
    # Insight Engine (Enterprise, Composite-First)
    # =====================================================
    
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates composite financial insights without
        thresholds, targets, or judgment language.
        """
    
        insights: List[Dict[str, Any]] = []
    
        # =================================================
        # Sub-domain: Revenue Quality (≥7)
        # =================================================
        if "revenue_total" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Revenue Scale Presence",
                    "so_what": (
                        "Revenue values are present at an observable scale, "
                        "providing a baseline view of operating magnitude."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue Distribution Spread",
                    "so_what": (
                        "Revenue values span a range of magnitudes, "
                        "indicating variation across observed periods."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue Central Tendency",
                    "so_what": (
                        "Average and median revenue levels offer a view "
                        "into typical operating performance."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue Extremes",
                    "so_what": (
                        "Minimum and maximum revenue observations highlight "
                        "the outer bounds of observed performance."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue Variability",
                    "so_what": (
                        "Revenue variability reflects how consistently "
                        "revenue levels change across observations."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue Range Dynamics",
                    "so_what": (
                        "The spread between revenue highs and lows "
                        "provides context on operational fluctuation."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue Signal Continuity",
                    "so_what": (
                        "The presence of sustained revenue signals "
                        "enables longitudinal financial interpretation."
                    )
                },
            ])
    
        # =================================================
        # Sub-domain: Cost Structure (≥7)
        # =================================================
        if "expense_total" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Expense Scale Presence",
                    "so_what": (
                        "Expense levels are observed alongside revenue, "
                        "indicating active cost structures."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Distribution Characteristics",
                    "so_what": (
                        "Expense values exhibit dispersion, "
                        "reflecting variation in cost behavior."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Central Tendency",
                    "so_what": (
                        "Average expense levels provide insight "
                        "into typical operating cost intensity."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Extremes",
                    "so_what": (
                        "Observed minimum and maximum expenses "
                        "frame the cost envelope."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Variability",
                    "so_what": (
                        "Expense variability highlights how costs "
                        "change across observations."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Range",
                    "so_what": (
                        "The range of expense values "
                        "illustrates cost fluctuation amplitude."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Signal Stability Context",
                    "so_what": (
                        "Consistent expense signals support "
                        "structural cost analysis."
                    )
                },
            ])
    
        # =================================================
        # Sub-domain: Profitability (≥7)
        # =================================================
        if "profit_total" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Profit Presence",
                    "so_what": (
                        "Profit values are observed, "
                        "indicating surplus after expenses."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Distribution Shape",
                    "so_what": (
                        "Profit values show dispersion, "
                        "revealing variation across observations."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Central Tendency",
                    "so_what": (
                        "Mean and median profit levels "
                        "represent typical earnings performance."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Extremes",
                    "so_what": (
                        "Observed profit highs and lows "
                        "define earnings boundaries."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Variability",
                    "so_what": (
                        "Profit variability reflects "
                        "earnings consistency over time."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Range Context",
                    "so_what": (
                        "The span between profit extremes "
                        "illustrates earnings fluctuation."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Relative to Revenue",
                    "so_what": (
                        "The relationship between profit and revenue "
                        "provides structural insight into margins."
                    )
                },
            ])
    
        # =================================================
        # Sub-domain: Liquidity & Capital Structure (≥7)
        # =================================================
        if "receivables_to_revenue_ratio" in kpis or "debt_to_assets_ratio" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Revenue Conversion to Receivables",
                    "so_what": (
                        "A portion of revenue manifests as receivables, "
                        "highlighting cash conversion dynamics."
                    )
                },
                {
                    "type": "composite",
                    "title": "Receivables Scale",
                    "so_what": (
                        "Receivable balances indicate the magnitude "
                        "of outstanding customer obligations."
                    )
                },
                {
                    "type": "composite",
                    "title": "Receivables Variability",
                    "so_what": (
                        "Variation in receivables reflects "
                        "changes in collection timing."
                    )
                },
                {
                    "type": "composite",
                    "title": "Debt Relative to Assets",
                    "so_what": (
                        "Debt levels are contextualized "
                        "against the asset base."
                    )
                },
                {
                    "type": "composite",
                    "title": "Capital Structure Balance",
                    "so_what": (
                        "Observed debt and asset relationships "
                        "frame capital composition."
                    )
                },
                {
                    "type": "composite",
                    "title": "Liquidity Signal Continuity",
                    "so_what": (
                        "Sustained liquidity proxies "
                        "support cash flow interpretation."
                    )
                },
                {
                    "type": "composite",
                    "title": "Balance Sheet Interplay",
                    "so_what": (
                        "Receivables, debt, and assets together "
                        "describe short- and long-term obligations."
                    )
                },
            ])
    
        # =================================================
        # Sub-domain: Financial Stability & Risk (≥7)
        # =================================================
        if "revenue_stability_std" in kpis or "expense_stability_std" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Revenue Stability Context",
                    "so_what": (
                        "Observed revenue dispersion informs "
                        "earnings predictability."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Stability Context",
                    "so_what": (
                        "Expense dispersion provides insight "
                        "into cost consistency."
                    )
                },
                {
                    "type": "composite",
                    "title": "Revenue vs Expense Stability",
                    "so_what": (
                        "Differences between revenue and expense stability "
                        "shape earnings variability."
                    )
                },
                {
                    "type": "composite",
                    "title": "Profit Stability",
                    "so_what": (
                        "Profit variability reflects combined "
                        "revenue and expense behavior."
                    )
                },
                {
                    "type": "composite",
                    "title": "Stability Range Signals",
                    "so_what": (
                        "Observed value ranges complement "
                        "dispersion measures."
                    )
                },
                {
                    "type": "composite",
                    "title": "Volatility Awareness",
                    "so_what": (
                        "Variability indicators support "
                        "risk-aware interpretation."
                    )
                },
                {
                    "type": "composite",
                    "title": "Temporal Consistency Signals",
                    "so_what": (
                        "Stability metrics enable longitudinal "
                        "financial assessment."
                    )
                },
            ])
    
        # =================================================
        # Sub-domain: Banking Health (≥7, Optional)
        # =================================================
        if "loans_mean" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Loan Portfolio Scale",
                    "so_what": (
                        "Loan balances indicate the size "
                        "of credit exposure."
                    )
                },
                {
                    "type": "composite",
                    "title": "Loan Distribution Characteristics",
                    "so_what": (
                        "Loan variability reflects "
                        "portfolio dispersion."
                    )
                },
                {
                    "type": "composite",
                    "title": "Non-Performing Asset Presence",
                    "so_what": (
                        "Non-performing assets coexist "
                        "within the loan portfolio."
                    )
                },
                {
                    "type": "composite",
                    "title": "Loan Quality Composition",
                    "so_what": (
                        "The relationship between loans and NPAs "
                        "describes portfolio composition."
                    )
                },
                {
                    "type": "composite",
                    "title": "Collateral Context",
                    "so_what": (
                        "Collateral levels provide "
                        "context for secured lending."
                    )
                },
                {
                    "type": "composite",
                    "title": "Credit Exposure Dynamics",
                    "so_what": (
                        "Loan behavior over time "
                        "frames credit dynamics."
                    )
                },
                {
                    "type": "composite",
                    "title": "Banking Signal Coverage",
                    "so_what": (
                        "Available banking signals support "
                        "credit portfolio interpretation."
                    )
                },
            ])
    
        # =================================================
        # Sub-domain: Integrity / Anomaly Signals (≥7)
        # =================================================
        if "expense_benford_deviation" in kpis:
            insights.extend([
                {
                    "type": "composite",
                    "title": "Expense Digit Distribution Pattern",
                    "so_what": (
                        "Expense values exhibit a measurable "
                        "leading-digit distribution pattern."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Value Diversity",
                    "so_what": (
                        "Expense entries show variation in value "
                        "and frequency."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Value Sign Behavior",
                    "so_what": (
                        "The presence of negative or zero expenses "
                        "adds context to ledger structure."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Entry Uniqueness",
                    "so_what": (
                        "The diversity of expense values "
                        "reflects recording patterns."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Distribution Shape",
                    "so_what": (
                        "Expense distributions reveal "
                        "concentration or dispersion."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Recording Consistency",
                    "so_what": (
                        "Observed expense patterns support "
                        "integrity interpretation."
                    )
                },
                {
                    "type": "composite",
                    "title": "Expense Signal Completeness",
                    "so_what": (
                        "Expense data coverage enables "
                        "pattern-based analysis."
                    )
                },
            ])
    
        # =================================================
        # Graceful Fallback
        # =================================================
        if not insights:
            insights.append({
                "type": "atomic",
                "title": "Financial Signal Availability",
                "so_what": (
                    "Available financial signals were limited, "
                    "so insights are constrained to observable data presence."
                )
            })
    
        return insights

    # =====================================================
    # Finance Domain — Block 6
    # Recommendation Engine (Enterprise, Advisory)
    # =====================================================
    
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates finance recommendations with:
        - Advisory language only
        - No prescriptions
        - No urgency or priority labels
        - Traceability to insight themes
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        insight_titles = [
            i.get("title", "").lower()
            for i in self.generate_insights(df, kpis)
        ]
    
        # =================================================
        # Sub-domain: Revenue Quality (≥7)
        # =================================================
        if any("revenue" in t for t in insight_titles):
            recommendations.extend([
                {
                    "recommendation": (
                        "Review how revenue levels evolve over time to better understand "
                        "the stability and scale of core operations."
                    ),
                    "related_area": "revenue_quality"
                },
                {
                    "recommendation": (
                        "Compare average and median revenue values to identify "
                        "potential skew or concentration effects."
                    ),
                    "related_area": "revenue_quality"
                },
                {
                    "recommendation": (
                        "Monitor revenue range and dispersion to understand "
                        "the breadth of operating outcomes."
                    ),
                    "related_area": "revenue_quality"
                },
                {
                    "recommendation": (
                        "Assess whether revenue variability aligns with "
                        "expected business seasonality or structural drivers."
                    ),
                    "related_area": "revenue_quality"
                },
                {
                    "recommendation": (
                        "Use revenue distribution patterns as a reference "
                        "when evaluating forecasting assumptions."
                    ),
                    "related_area": "revenue_quality"
                },
                {
                    "recommendation": (
                        "Observe revenue continuity over time to support "
                        "longitudinal financial interpretation."
                    ),
                    "related_area": "revenue_quality"
                },
                {
                    "recommendation": (
                        "Contextualize revenue magnitude alongside other "
                        "financial signals to understand overall scale."
                    ),
                    "related_area": "revenue_quality"
                },
            ])
    
        # =================================================
        # Sub-domain: Cost Structure (≥7)
        # =================================================
        if any("expense" in t for t in insight_titles):
            recommendations.extend([
                {
                    "recommendation": (
                        "Review expense distributions to understand "
                        "how costs vary across observations."
                    ),
                    "related_area": "cost_structure"
                },
                {
                    "recommendation": (
                        "Compare average and extreme expense values "
                        "to assess cost envelope breadth."
                    ),
                    "related_area": "cost_structure"
                },
                {
                    "recommendation": (
                        "Monitor expense variability to evaluate "
                        "cost flexibility under changing conditions."
                    ),
                    "related_area": "cost_structure"
                },
                {
                    "recommendation": (
                        "Observe how expense behavior coexists "
                        "with revenue movements over time."
                    ),
                    "related_area": "cost_structure"
                },
                {
                    "recommendation": (
                        "Use expense range signals to inform "
                        "scenario planning discussions."
                    ),
                    "related_area": "cost_structure"
                },
                {
                    "recommendation": (
                        "Review cost concentration patterns "
                        "to understand structural expense drivers."
                    ),
                    "related_area": "cost_structure"
                },
                {
                    "recommendation": (
                        "Track expense continuity to support "
                        "consistent financial analysis."
                    ),
                    "related_area": "cost_structure"
                },
            ])
    
        # =================================================
        # Sub-domain: Profitability (≥7)
        # =================================================
        if any("profit" in t for t in insight_titles):
            recommendations.extend([
                {
                    "recommendation": (
                        "Review profit distributions to understand "
                        "earnings variability across observations."
                    ),
                    "related_area": "profitability"
                },
                {
                    "recommendation": (
                        "Compare profit central tendency measures "
                        "to evaluate typical earnings performance."
                    ),
                    "related_area": "profitability"
                },
                {
                    "recommendation": (
                        "Observe profit extremes to contextualize "
                        "earnings boundaries."
                    ),
                    "related_area": "profitability"
                },
                {
                    "recommendation": (
                        "Monitor profit variability to support "
                        "earnings stability discussions."
                    ),
                    "related_area": "profitability"
                },
                {
                    "recommendation": (
                        "Examine profit behavior alongside "
                        "revenue and expense signals."
                    ),
                    "related_area": "profitability"
                },
                {
                    "recommendation": (
                        "Use profit-to-revenue relationships "
                        "as structural context for margin analysis."
                    ),
                    "related_area": "profitability"
                },
                {
                    "recommendation": (
                        "Review longitudinal profit patterns "
                        "to support planning assumptions."
                    ),
                    "related_area": "profitability"
                },
            ])
    
        # =================================================
        # Sub-domain: Liquidity & Capital Structure (≥7)
        # =================================================
        if any("receivable" in t or "debt" in t or "asset" in t for t in insight_titles):
            recommendations.extend([
                {
                    "recommendation": (
                        "Review receivables behavior alongside revenue "
                        "to better understand cash conversion dynamics."
                    ),
                    "related_area": "liquidity"
                },
                {
                    "recommendation": (
                        "Monitor receivable variability to assess "
                        "collection pattern consistency."
                    ),
                    "related_area": "liquidity"
                },
                {
                    "recommendation": (
                        "Assess debt levels relative to assets "
                        "to understand capital composition."
                    ),
                    "related_area": "capital_structure"
                },
                {
                    "recommendation": (
                        "Observe how debt and asset signals evolve "
                        "together over time."
                    ),
                    "related_area": "capital_structure"
                },
                {
                    "recommendation": (
                        "Use balance sheet relationships as context "
                        "for funding discussions."
                    ),
                    "related_area": "capital_structure"
                },
                {
                    "recommendation": (
                        "Review liquidity proxy continuity "
                        "to support cash flow interpretation."
                    ),
                    "related_area": "liquidity"
                },
                {
                    "recommendation": (
                        "Contextualize short- and long-term obligations "
                        "within the broader financial structure."
                    ),
                    "related_area": "capital_structure"
                },
            ])
    
        # =================================================
        # Sub-domain: Banking Health (≥7, Optional)
        # =================================================
        if any("loan" in t or "bank" in t for t in insight_titles):
            recommendations.extend([
                {
                    "recommendation": (
                        "Review loan portfolio size and distribution "
                        "to understand credit exposure scale."
                    ),
                    "related_area": "banking"
                },
                {
                    "recommendation": (
                        "Monitor loan variability to assess "
                        "portfolio dispersion."
                    ),
                    "related_area": "banking"
                },
                {
                    "recommendation": (
                        "Observe non-performing asset levels "
                        "alongside total loans for composition context."
                    ),
                    "related_area": "banking"
                },
                {
                    "recommendation": (
                        "Review collateral signals to contextualize "
                        "secured lending exposure."
                    ),
                    "related_area": "banking"
                },
                {
                    "recommendation": (
                        "Track loan and collateral dynamics "
                        "over time to understand credit behavior."
                    ),
                    "related_area": "banking"
                },
                {
                    "recommendation": (
                        "Use banking signal continuity "
                        "to support portfolio interpretation."
                    ),
                    "related_area": "banking"
                },
                {
                    "recommendation": (
                        "Contextualize banking metrics "
                        "within overall financial structure."
                    ),
                    "related_area": "banking"
                },
            ])
    
        # =================================================
        # Sub-domain: Integrity / Data Quality (≥7)
        # =================================================
        if any("expense" in t or "digit" in t for t in insight_titles):
            recommendations.extend([
                {
                    "recommendation": (
                        "Review expense data recording practices "
                        "to understand value distribution patterns."
                    ),
                    "related_area": "data_integrity"
                },
                {
                    "recommendation": (
                        "Monitor diversity of expense values "
                        "to assess recording consistency."
                    ),
                    "related_area": "data_integrity"
                },
                {
                    "recommendation": (
                        "Observe zero or negative expense entries "
                        "for contextual interpretation."
                    ),
                    "related_area": "data_integrity"
                },
                {
                    "recommendation": (
                        "Compare expense distribution shapes "
                        "across periods for consistency."
                    ),
                    "related_area": "data_integrity"
                },
                {
                    "recommendation": (
                        "Use digit-distribution patterns "
                        "as contextual integrity signals."
                    ),
                    "related_area": "data_integrity"
                },
                {
                    "recommendation": (
                        "Review aggregation logic for expenses "
                        "to ensure uniform treatment."
                    ),
                    "related_area": "data_integrity"
                },
                {
                    "recommendation": (
                        "Interpret expense anomaly signals "
                        "alongside other financial indicators."
                    ),
                    "related_area": "data_integrity"
                },
            ])
    
        # =================================================
        # Graceful Fallback
        # =================================================
        if not recommendations:
            recommendations.append({
                "recommendation": (
                    "Continue monitoring available financial signals "
                    "to expand analytical depth as data coverage improves."
                ),
                "related_area": "general"
            })
    
        return recommendations

# =====================================================
# Finance Domain — Block 7
# Domain Detector (Boundary-Safe, Enterprise)
# =====================================================

class FinanceDomainDetector(BaseDomainDetector):
    """
    Boundary-safe detector for the Finance domain.
    Prevents collision with Retail, Marketing,
    Supply Chain, and generic transactional datasets.
    """

    domain_name = "finance"

    # Token clusters (semantic, not mandatory)
    PNL_TOKENS = {
        "revenue", "sales", "income",
        "expense", "cost", "cogs", "opex",
        "profit", "ebit", "ebitda"
    }

    BALANCE_SHEET_TOKENS = {
        "asset", "assets",
        "liability", "liabilities",
        "equity", "debt"
    }

    FINANCE_OPS_TOKENS = {
        "loan", "interest",
        "receivable", "payable",
        "ledger", "npa"
    }

    EXCLUDED_GENERIC_TOKENS = {
        "price", "volume", "date", "id"
    }

    # -------------------------------------------------
    # Detection Logic
    # -------------------------------------------------

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        cols = [str(c).lower() for c in df.columns]

        def tokenize(col: str) -> set:
            return set(col.replace("_", " ").split())

        tokenized_cols = {c: tokenize(c) for c in cols}

        pnl_hits = {
            c for c, toks in tokenized_cols.items()
            if toks & self.PNL_TOKENS
        }

        bs_hits = {
            c for c, toks in tokenized_cols.items()
            if toks & self.BALANCE_SHEET_TOKENS
        }

        ops_hits = {
            c for c, toks in tokenized_cols.items()
            if toks & self.FINANCE_OPS_TOKENS
        }

        generic_hits = {
            c for c, toks in tokenized_cols.items()
            if toks & self.EXCLUDED_GENERIC_TOKENS
        }

        # -------------------------------------------------
        # Numeric Signal Validation
        # -------------------------------------------------
        numeric_finance_cols = {
            c for c in (pnl_hits | bs_hits | ops_hits)
            if pd.api.types.is_numeric_dtype(df[c])
        }

        # -------------------------------------------------
        # Confidence Construction (Structural, Not Threshold)
        # -------------------------------------------------
        signal_groups_present = sum([
            bool(pnl_hits),
            bool(bs_hits),
            bool(ops_hits)
        ])

        confidence = signal_groups_present / 3.0 if numeric_finance_cols else 0.0

        # Suppress generic-only datasets
        if signal_groups_present == 1 and generic_hits:
            confidence *= 0.4

        # No finance semantics
        if signal_groups_present == 0:
            confidence = 0.0

        has_gl = any("gl" in c and "account" in c for c in cols)
        has_pnl = bool(pnl_hits)
        
        if has_gl and has_pnl:
            confidence = max(confidence, 0.85)
            
        return DomainDetectionResult(
            domain="finance",
            confidence=round(confidence, 2),
            metadata={
                "pnl_columns": sorted(pnl_hits),
                "balance_sheet_columns": sorted(bs_hits),
                "finance_ops_columns": sorted(ops_hits),
                "numeric_finance_columns": sorted(numeric_finance_cols),
                "generic_columns": sorted(generic_hits),
            }
        )


# =====================================================
# Registration (Framework-Consistent)
# =====================================================

def register(registry):
    registry.register(
        "finance",
        FinanceDomain,
        FinanceDomainDetector
    )
