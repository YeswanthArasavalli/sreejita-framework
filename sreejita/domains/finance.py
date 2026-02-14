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
        "reporting_date",
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
        coverage_periods=int(series.nunique(dropna=True)),
    )


# =====================================================
# FINANCE ANALYTIC HELPERS (OPTIONAL SIGNALS)
# =====================================================

def benford_deviation(series: pd.Series) -> float:
    """
    Measures deviation from expected leading-digit distribution.
    Weak integrity signal only (not fraud detection).
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


def _has_numeric_column(df: pd.DataFrame, names: List[str]) -> bool:
    names = {n.lower() for n in names}
    for col in df.columns:
        lcol = str(col).lower()
        if any(n in lcol for n in names) and pd.api.types.is_numeric_dtype(df[col]):
            return True
    return False


def _has_gl_account(df: pd.DataFrame) -> bool:
    for col in df.columns:
        l = str(col).lower()
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
# FINANCE DOMAIN — FOUNDATION & PREPROCESS
# =====================================================

class FinanceDomain(BaseDomain):
    name = "finance"
    description = "Universal Finance Intelligence (Health, Efficiency, Risk)"

    # -------------------------------------------------
    # PREPROCESS (STRICTLY GOVERNED)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finance preprocess guarantees:
        - No fabrication
        - No assumptions
        - Signal-driven only
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("FinanceDomain.preprocess expects a DataFrame")

        df = df.copy(deep=False)

        # -----------------------------
        # TIME CONTEXT
        # -----------------------------
        self.time_context = build_time_context(df)
        self.time_col = self.time_context.time_column

        # -----------------------------
        # COLUMN RESOLUTION (SOFT)
        # -----------------------------
        def _resolve_any(candidates: List[str]) -> Optional[str]:
            for c in candidates:
                col = resolve_column(df, c)
                if col:
                    return col
            return None

        self.cols: Dict[str, Optional[str]] = {
            # Income Statement
            "revenue": _resolve_any(["revenue", "sales", "turnover"]),
            "expense": _resolve_any(["expense", "cost", "opex", "cogs"]),
            "profit": _resolve_any(["profit", "net_income", "ebit", "ebitda"]),

            # Balance Sheet
            "assets": _resolve_any(["assets", "total_assets"]),
            "equity": _resolve_any(["equity", "shareholder_equity"]),
            "debt": _resolve_any(["debt", "liabilities", "total_debt"]),

            # Credit / Banking (Optional)
            "receivables": _resolve_any(["accounts_receivable", "receivables"]),
            "loans": _resolve_any(["loan_amount", "loans"]),
            "npa": _resolve_any(["non_performing_assets", "npa"]),
            "collateral": _resolve_any(["collateral_value"]),
            "interest": _resolve_any(["interest_expense", "interest"]),

            # Market Data (Optional)
            "close": _resolve_any(["close", "adj_close", "price"]),
            "volume": _resolve_any(["volume"]),
        }

        # -----------------------------
        # AVAILABLE SIGNAL REGISTRY
        # -----------------------------
        self.available_signals = {
            k: v for k, v in self.cols.items() if v is not None
        }

        # -----------------------------
        # NUMERIC NORMALIZATION
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
        # TIME ORDERING
        # -----------------------------
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col).reset_index(drop=True)

        return df

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

        if not isinstance(df, pd.DataFrame):
            raise TypeError("FinanceDomain.preprocess expects a DataFrame")

        df = df.copy(deep=False)

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

        self.cols: Dict[str, Optional[str]] = {
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
        # -----------------------------
        self.available_signals: Dict[str, str] = {
            k: v for k, v in self.cols.items() if v is not None
        }

        # -----------------------------
        # Numeric Safety (No Fabrication)
        # -----------------------------
        for col in self.available_signals.values():
            if col in df.columns and df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[,$]", "", regex=True)
                    .replace("", np.nan)
                )
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -----------------------------
        # Time Ordering (If Possible)
        # -----------------------------
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col], errors="coerce"
            )
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

            kpis = self.calculate_kpis(df)
            if not isinstance(kpis, dict):
                kpis = {}

            if "_confidence" not in kpis:
                kpis["_confidence"] = {}

            self._last_kpis = kpis

            insights = self.generate_insights(df, kpis) or []
            recommendations = self.generate_recommendations(df, kpis) or []

            visuals: List[Dict[str, Any]] = []
            if visual_output_dir is not None:
                try:
                    visuals = self.generate_visuals(df, visual_output_dir)
                    visuals = self.ensure_minimum_visuals(
                        visuals, df, visual_output_dir
                    )
                except Exception:
                    visuals = self.ensure_minimum_visuals(
                        [], df, visual_output_dir
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
        Finance KPI Engine (v1.0)
    
        GUARANTEES:
        - Capability-driven sub-domains
        - No thresholds or targets
        - No fabricated KPIs
        - Graceful degradation
        - Confidence-tagged KPIs
        """
    
        if df is None or df.empty:
            return {}
    
        c = self.available_signals
        time_col = self.time_col
        volume = int(len(df))
    
        # -------------------------------------------------
        # SUB-DOMAIN DEFINITIONS (LOCKED)
        # -------------------------------------------------
        sub_domains = {
            "revenue": "Revenue Quality",
            "cost": "Cost Structure",
            "profitability": "Profitability",
            "liquidity": "Liquidity & Cash Proxies",
            "stability": "Financial Stability & Risk",
            "banking": "Banking Health",
            "market": "Market Variability",
            "integrity": "Integrity & Anomaly Signals",
        }
    
        # -------------------------------------------------
        # KPI CONTAINER
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "domain": "finance",
            "sub_domains": sub_domains,
            "record_count": volume,
            "_domain_kpi_map": {},
            "_confidence": {},
        }
    
        # -------------------------------------------------
        # SAFE HELPERS
        # -------------------------------------------------
        def safe_series(col):
            if not col or col not in df.columns:
                return None
            s = df[col].dropna()
            return s if not s.empty else None
    
        # =================================================
        # REVENUE — QUALITY (≥7)
        # =================================================
        revenue = []
        s = safe_series(c.get("revenue"))
        if s is not None:
            kpis.update({
                "revenue_total": s.sum(),
                "revenue_mean": s.mean(),
                "revenue_median": s.median(),
                "revenue_min": s.min(),
                "revenue_max": s.max(),
                "revenue_variability": s.std(),
                "revenue_range": s.max() - s.min(),
            })
            revenue.extend([
                "revenue_total",
                "revenue_mean",
                "revenue_median",
                "revenue_min",
                "revenue_max",
                "revenue_variability",
                "revenue_range",
            ])
    
        # =================================================
        # COST — STRUCTURE (≥7)
        # =================================================
        cost = []
        s = safe_series(c.get("expense"))
        if s is not None:
            kpis.update({
                "expense_total": s.sum(),
                "expense_mean": s.mean(),
                "expense_median": s.median(),
                "expense_min": s.min(),
                "expense_max": s.max(),
                "expense_variability": s.std(),
                "expense_range": s.max() - s.min(),
            })
            cost.extend([
                "expense_total",
                "expense_mean",
                "expense_median",
                "expense_min",
                "expense_max",
                "expense_variability",
                "expense_range",
            ])
    
        # =================================================
        # PROFITABILITY (≥7)
        # =================================================
        profitability = []
        s = safe_series(c.get("profit"))
        if s is not None:
            kpis.update({
                "profit_total": s.sum(),
                "profit_mean": s.mean(),
                "profit_median": s.median(),
                "profit_min": s.min(),
                "profit_max": s.max(),
                "profit_variability": s.std(),
                "profit_range": s.max() - s.min(),
            })
            profitability.extend([
                "profit_total",
                "profit_mean",
                "profit_median",
                "profit_min",
                "profit_max",
                "profit_variability",
                "profit_range",
            ])
    
            if c.get("revenue"):
                rev_mean = df[c["revenue"]].mean()
                if pd.notna(rev_mean):
                    kpis["profit_to_revenue_ratio"] = safe_divide(
                        s.mean(), rev_mean
                    )
                    profitability.append("profit_to_revenue_ratio")
    
        # =================================================
        # LIQUIDITY & CASH PROXIES
        # =================================================
        liquidity = []
    
        s = safe_series(c.get("receivables"))
        if s is not None:
            kpis.update({
                "receivables_mean": s.mean(),
                "receivables_median": s.median(),
                "receivables_max": s.max(),
                "receivables_variability": s.std(),
            })
            liquidity.extend([
                "receivables_mean",
                "receivables_median",
                "receivables_max",
                "receivables_variability",
            ])
    
        if c.get("receivables") and c.get("revenue"):
            kpis["receivables_to_revenue_ratio"] = safe_divide(
                df[c["receivables"]].mean(),
                df[c["revenue"]].mean(),
            )
            liquidity.append("receivables_to_revenue_ratio")
    
        if c.get("assets") and c.get("debt"):
            kpis["debt_to_assets_ratio"] = safe_divide(
                df[c["debt"]].mean(),
                df[c["assets"]].mean(),
            )
            liquidity.append("debt_to_assets_ratio")
    
        # =================================================
        # STABILITY & RISK
        # =================================================
        stability = []
    
        for key, label in [("revenue", "revenue"), ("expense", "expense"), ("profit", "profit")]:
            s = safe_series(c.get(key))
            if s is not None and len(s) > 2:
                kpis[f"{label}_stability_std"] = s.std()
                kpis[f"{label}_stability_range"] = s.max() - s.min()
                stability.extend([
                    f"{label}_stability_std",
                    f"{label}_stability_range",
                ])
    
        # =================================================
        # BANKING HEALTH (OPTIONAL)
        # =================================================
        banking = []
    
        s = safe_series(c.get("loans"))
        if s is not None:
            kpis.update({
                "loans_mean": s.mean(),
                "loans_max": s.max(),
                "loans_variability": s.std(),
            })
            banking.extend([
                "loans_mean",
                "loans_max",
                "loans_variability",
            ])
    
        if c.get("npa") and c.get("loans"):
            kpis["npa_to_loans_ratio"] = safe_divide(
                df[c["npa"]].mean(),
                df[c["loans"]].mean(),
            )
            banking.append("npa_to_loans_ratio")
    
        if c.get("collateral") and c.get("loans"):
            kpis["loan_to_collateral_ratio"] = safe_divide(
                df[c["loans"]].mean(),
                df[c["collateral"]].mean(),
            )
            banking.append("loan_to_collateral_ratio")
    
        # =================================================
        # MARKET VARIABILITY (OPTIONAL)
        # =================================================
        market = []
    
        if c.get("close") and time_col:
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
                    market.extend([
                        "price_return_mean",
                        "price_return_std",
                        "price_return_min",
                        "price_return_max",
                        "price_level_mean",
                        "price_level_max",
                        "price_level_min",
                    ])
    
        # =================================================
        # INTEGRITY & ANOMALY SIGNALS
        # =================================================
        integrity = []
    
        s = safe_series(c.get("expense"))
        if s is not None:
            kpis.update({
                "expense_benford_deviation": benford_deviation(s),
                "expense_unique_ratio": s.nunique() / len(s),
                "expense_zero_ratio": (s == 0).mean(),
                "expense_negative_ratio": (s < 0).mean(),
            })
            integrity.extend([
                "expense_benford_deviation",
                "expense_unique_ratio",
                "expense_zero_ratio",
                "expense_negative_ratio",
            ])
    
        # -------------------------------------------------
        # DOMAIN → KPI MAP
        # -------------------------------------------------
        kpis["_domain_kpi_map"] = {
            "revenue": revenue,
            "cost": cost,
            "profitability": profitability,
            "liquidity": liquidity,
            "stability": stability,
            "banking": banking,
            "market": market,
            "integrity": integrity,
        }
    
        # -------------------------------------------------
        # KPI CONFIDENCE (MANDATORY)
        # -------------------------------------------------
        for key, val in kpis.items():
            if key.startswith("_") or not isinstance(val, (int, float, np.floating)):
                continue
    
            base = 0.75
            if volume < 100:
                base -= 0.15
            if "ratio" in key or "return" in key:
                base -= 0.05
            if "benford" in key:
                base -= 0.10
    
            kpis["_confidence"][key] = round(
                max(0.4, min(0.9, base)), 2
            )
    
        self._last_kpis = kpis
        return kpis


    # =====================================================
    # Finance Domain — Block 4
    # Visual Engine (Enterprise, Governed)
    # =====================================================
    
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Finance Visual Engine (v1.0)
    
        GUARANTEES:
        - Observational only (no targets, no thresholds)
        - KPI-evidence aligned
        - Sub-domain governed
        - Deterministic output
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.available_signals
        time_col = self.time_col
        record_count = int(len(df))
    
        # -------------------------------------------------
        # VISUAL CONFIDENCE (DATA VOLUME ONLY)
        # -------------------------------------------------
        if record_count >= 5000:
            visual_conf = 0.85
        elif record_count >= 1000:
            visual_conf = 0.7
        else:
            visual_conf = 0.55
    
        # -------------------------------------------------
        # SAVE HELPER (STANDARDIZED)
        # -------------------------------------------------
        def save(fig, name, caption, importance, sub_domain, role, axis):
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "sub_domain": sub_domain,
                "role": role,
                "axis": axis,
                "confidence": visual_conf,
            })
    
        # -------------------------------------------------
        # FORMATTERS
        # -------------------------------------------------
        def human_fmt(x, _):
            if pd.isna(x):
                return ""
            x = float(x)
            if abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return f"{x:.0f}"
    
        # =================================================
        # MARKET VARIABILITY
        # =================================================
        if c.get("close") and time_col:
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
                save(
                    fig,
                    "market_price_trend.png",
                    "Observed price movement over time",
                    0.95,
                    "market",
                    "trend",
                    "time",
                )
    
                fig, ax = plt.subplots(figsize=(6, 4))
                s.plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Price Level Distribution")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "market_price_distribution.png",
                    "Observed distribution of price levels",
                    0.80,
                    "market",
                    "distribution",
                    "value",
                )
    
                returns = s.pct_change().dropna()
                if not returns.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    returns.plot(kind="hist", bins=20, ax=ax)
                    ax.set_title("Price Return Distribution")
                    save(
                        fig,
                        "market_return_distribution.png",
                        "Observed distribution of price changes",
                        0.78,
                        "market",
                        "distribution",
                        "return",
                    )
    
        # =================================================
        # REVENUE & COST STRUCTURE
        # =================================================
        if c.get("revenue"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["revenue"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Revenue Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "corp_revenue_distribution.png",
                "Observed distribution of revenue values",
                0.88,
                "revenue",
                "distribution",
                "value",
            )
    
        if c.get("expense"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["expense"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Expense Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "corp_expense_distribution.png",
                "Observed distribution of expense values",
                0.86,
                "cost",
                "distribution",
                "value",
            )
    
        if c.get("revenue") and c.get("expense"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Revenue", "Expense"],
                [df[c["revenue"]].sum(), df[c["expense"]].sum()]
            )
            ax.set_title("Revenue vs Expense Magnitude")
            ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
            save(
                fig,
                "corp_revenue_expense.png",
                "Relative magnitude of revenue and expenses",
                0.92,
                "profitability",
                "comparison",
                "aggregate",
            )
    
        # =================================================
        # PROFITABILITY
        # =================================================
        if c.get("profit"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["profit"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Profit Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "corp_profit_distribution.png",
                "Observed distribution of profit values",
                0.84,
                "profitability",
                "distribution",
                "value",
            )
    
        # =================================================
        # CAPITAL STRUCTURE
        # =================================================
        if c.get("debt") and c.get("equity"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Debt", "Equity"],
                [df[c["debt"]].mean(), df[c["equity"]].mean()]
            )
            ax.set_title("Average Capital Structure")
            ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
            save(
                fig,
                "risk_capital_structure.png",
                "Observed capital structure components",
                0.83,
                "liquidity",
                "structure",
                "average",
            )
    
        # =================================================
        # BANKING HEALTH (OPTIONAL)
        # =================================================
        if c.get("loans"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["loans"]].dropna().plot(kind="hist", bins=20, ax=ax)
            ax.set_title("Loan Value Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(
                fig,
                "banking_loan_distribution.png",
                "Observed distribution of loan values",
                0.80,
                "banking",
                "distribution",
                "value",
            )
    
        if c.get("loans") and c.get("npa"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                ["Loans", "Non-Performing Assets"],
                [df[c["loans"]].mean(), df[c["npa"]].mean()]
            )
            ax.set_title("Loans vs Non-Performing Assets")
            ax.yaxis.set_major_formatter(FuncFormatter(human_currency_formatter))
            save(
                fig,
                "banking_loan_npa.png",
                "Observed loan and non-performing asset levels",
                0.78,
                "banking",
                "comparison",
                "average",
            )
    
        # =================================================
        # INTEGRITY / ANOMALY SIGNALS
        # =================================================
        if c.get("expense"):
            dev = benford_deviation(df[c["expense"]])
            if dev > 0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(["Deviation"], [dev])
                ax.set_title("Expense Digit Distribution Deviation")
                save(
                    fig,
                    "integrity_benford_expense.png",
                    "Observed deviation in leading digit distribution",
                    0.75,
                    "integrity",
                    "integrity",
                    "index",
                )
    
        # -------------------------------------------------
        # MANY → FEW (REPORT LAYER DECIDES)
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals

    # =====================================================
    # Finance Domain — Block 5
    # Insight Engine (Enterprise, Composite-First)
    # =====================================================
    
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Finance Composite Insight Engine (v1.0)
    
        GUARANTEES:
        - Composite-first insights
        - ≥7 insights per sub-domain (when data allows)
        - Observational, executive-safe language
        - No thresholds, no targets, no judgement
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        # =================================================
        # REVENUE — SCALE & DISTRIBUTION
        # =================================================
        if "revenue_total" in kpis:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Scale Observed",
                    "so_what": "Revenue values provide a clear indication of operating scale."
                },
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Distribution Spread",
                    "so_what": "Revenue observations span a range of magnitudes across periods."
                },
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Central Tendency",
                    "so_what": "Average and median revenue values describe typical operating levels."
                },
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Extremes Context",
                    "so_what": "Minimum and maximum revenue values define observed boundaries."
                },
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Variability Signal",
                    "so_what": "Revenue variability reflects changes across reporting periods."
                },
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Range Dynamics",
                    "so_what": "The spread between revenue highs and lows provides fluctuation context."
                },
                {
                    "level": "INFO",
                    "sub_domain": "revenue",
                    "title": "Revenue Continuity Signal",
                    "so_what": "Sustained revenue presence enables longitudinal interpretation."
                },
            ])
    
        # =================================================
        # COST — STRUCTURE & BEHAVIOR
        # =================================================
        if "expense_total" in kpis:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Expense Scale Observed",
                    "so_what": "Expense values indicate the presence of an operating cost structure."
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Expense Distribution Characteristics",
                    "so_what": "Expense values show dispersion across observed periods."
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Expense Central Tendency",
                    "so_what": "Average expense levels describe typical cost intensity."
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Expense Extremes Context",
                    "so_what": "Minimum and maximum expenses frame cost boundaries."
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Expense Variability Signal",
                    "so_what": "Expense variability reflects how costs change over time."
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Expense Range Dynamics",
                    "so_what": "The range of expense values illustrates cost fluctuation amplitude."
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Signal Continuity",
                    "so_what": "Consistent expense signals support structural cost analysis."
                },
            ])
    
        # =================================================
        # PROFITABILITY — EARNINGS STRUCTURE
        # =================================================
        if "profit_total" in kpis:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit Presence Observed",
                    "so_what": "Profit values indicate surplus after operating costs."
                },
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit Distribution Shape",
                    "so_what": "Profit dispersion reflects variability across periods."
                },
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit Central Tendency",
                    "so_what": "Mean and median profit describe typical earnings performance."
                },
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit Extremes Context",
                    "so_what": "Observed profit highs and lows define earnings bounds."
                },
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit Variability Signal",
                    "so_what": "Profit variability reflects combined revenue and cost behavior."
                },
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit Range Dynamics",
                    "so_what": "The span between profit extremes illustrates earnings fluctuation."
                },
                {
                    "level": "INFO",
                    "sub_domain": "profitability",
                    "title": "Profit–Revenue Relationship",
                    "so_what": "The relationship between profit and revenue provides margin context."
                },
            ])
    
        # =================================================
        # LIQUIDITY & CAPITAL STRUCTURE
        # =================================================
        if (
            "receivables_to_revenue_ratio" in kpis
            or "debt_to_assets_ratio" in kpis
        ):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Receivables Conversion Context",
                    "so_what": "A portion of revenue appears as receivables, shaping cash conversion."
                },
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Receivables Scale Observed",
                    "so_what": "Receivable balances indicate outstanding obligations."
                },
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Receivables Variability",
                    "so_what": "Variation in receivables reflects collection timing dynamics."
                },
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Debt Relative to Assets",
                    "so_what": "Debt levels are contextualized against the asset base."
                },
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Capital Structure Balance",
                    "so_what": "Observed debt and assets describe capital composition."
                },
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Liquidity Signal Continuity",
                    "so_what": "Sustained liquidity proxies enable cash flow interpretation."
                },
                {
                    "level": "INFO",
                    "sub_domain": "liquidity",
                    "title": "Balance Sheet Interplay",
                    "so_what": "Receivables, debt, and assets jointly describe obligations."
                },
            ])
    
        # =================================================
        # STABILITY & RISK
        # =================================================
        if (
            "revenue_stability_std" in kpis
            or "expense_stability_std" in kpis
            or "profit_stability_std" in kpis
        ):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Revenue Stability Context",
                    "so_what": "Revenue dispersion informs earnings consistency."
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Expense Stability Context",
                    "so_what": "Expense dispersion provides insight into cost consistency."
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Revenue vs Expense Stability",
                    "so_what": "Differences between revenue and cost stability shape earnings variability."
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Profit Stability Signal",
                    "so_what": "Profit variability reflects combined operating dynamics."
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Range-Based Risk Signals",
                    "so_what": "Observed value ranges complement dispersion measures."
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Volatility Awareness",
                    "so_what": "Variability indicators support risk-aware interpretation."
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Temporal Consistency Signals",
                    "so_what": "Stability metrics enable longitudinal financial assessment."
                },
            ])
    
        # =================================================
        # INTEGRITY / ANOMALY SIGNALS
        # =================================================
        if "expense_benford_deviation" in kpis:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Digit Distribution Pattern",
                    "so_what": "Expense values exhibit a measurable leading-digit distribution pattern."
                },
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Value Diversity",
                    "so_what": "Expense entries show variation in magnitude and frequency."
                },
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Sign Behavior",
                    "so_what": "Negative or zero expenses add structural context."
                },
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Entry Uniqueness",
                    "so_what": "The diversity of expense values reflects recording patterns."
                },
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Distribution Shape",
                    "so_what": "Expense distributions reveal concentration or dispersion."
                },
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Recording Consistency",
                    "so_what": "Observed expense patterns support integrity interpretation."
                },
                {
                    "level": "INFO",
                    "sub_domain": "integrity",
                    "title": "Expense Signal Coverage",
                    "so_what": "Expense data coverage enables pattern-based analysis."
                },
            ])
    
        # =================================================
        # GUARANTEED FALLBACK
        # =================================================
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Financial Signal Availability",
                "so_what": "Available financial signals provide limited but observable context."
            })
    
        return insights


    # =====================================================
    # Finance Domain — Block 6
    # Recommendation Engine (Enterprise, Advisory)
    # =====================================================
    
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Finance Recommendation Engine (v1.0)
    
        GUARANTEES:
        - Advisory-only language
        - No urgency, no prioritization
        - Signal-gated (no blind advice)
        - Sub-domain aligned
        - Deterministic output
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return recommendations
    
        # =================================================
        # REVENUE — QUALITY & SCALE
        # =================================================
        if "revenue_total" in kpis:
            recommendations.extend([
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Review how revenue levels evolve over time to understand "
                        "the scale and continuity of core operations."
                    ),
                },
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Compare average and median revenue values to identify "
                        "potential skew or concentration effects."
                    ),
                },
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Observe revenue variability and range to contextualize "
                        "operational fluctuation."
                    ),
                },
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Assess whether observed revenue patterns align with "
                        "known business cycles or structural drivers."
                    ),
                },
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Use revenue distribution characteristics as context "
                        "for forecasting and planning discussions."
                    ),
                },
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Monitor revenue continuity to support longitudinal "
                        "financial interpretation."
                    ),
                },
                {
                    "sub_domain": "revenue",
                    "action": (
                        "Interpret revenue magnitude alongside expense and "
                        "profit signals to understand overall scale."
                    ),
                },
            ])
    
        # =================================================
        # COST — STRUCTURE & BEHAVIOR
        # =================================================
        if "expense_total" in kpis:
            recommendations.extend([
                {
                    "sub_domain": "cost",
                    "action": (
                        "Review expense distributions to understand how "
                        "costs vary across observations."
                    ),
                },
                {
                    "sub_domain": "cost",
                    "action": (
                        "Compare typical and extreme expense values to "
                        "frame the operating cost envelope."
                    ),
                },
                {
                    "sub_domain": "cost",
                    "action": (
                        "Monitor expense variability to understand "
                        "cost flexibility under changing conditions."
                    ),
                },
                {
                    "sub_domain": "cost",
                    "action": (
                        "Observe how expense behavior coexists with "
                        "revenue movements over time."
                    ),
                },
                {
                    "sub_domain": "cost",
                    "action": (
                        "Use expense range signals as context "
                        "for scenario discussions."
                    ),
                },
                {
                    "sub_domain": "cost",
                    "action": (
                        "Review expense concentration patterns to "
                        "understand structural cost drivers."
                    ),
                },
                {
                    "sub_domain": "cost",
                    "action": (
                        "Track expense continuity to support "
                        "consistent financial analysis."
                    ),
                },
            ])
    
        # =================================================
        # PROFITABILITY — EARNINGS STRUCTURE
        # =================================================
        if "profit_total" in kpis:
            recommendations.extend([
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Review profit distributions to understand "
                        "earnings variability across periods."
                    ),
                },
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Compare central tendency measures to "
                        "contextualize typical earnings performance."
                    ),
                },
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Observe profit extremes to frame "
                        "earnings boundaries."
                    ),
                },
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Monitor profit variability alongside "
                        "revenue and expense behavior."
                    ),
                },
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Use profit-to-revenue relationships "
                        "as structural margin context."
                    ),
                },
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Review longitudinal profit patterns "
                        "to support planning assumptions."
                    ),
                },
                {
                    "sub_domain": "profitability",
                    "action": (
                        "Interpret profitability signals "
                        "within the broader financial structure."
                    ),
                },
            ])
    
        # =================================================
        # LIQUIDITY & CAPITAL STRUCTURE
        # =================================================
        if (
            "receivables_to_revenue_ratio" in kpis
            or "debt_to_assets_ratio" in kpis
        ):
            recommendations.extend([
                {
                    "sub_domain": "liquidity",
                    "action": (
                        "Review receivables behavior alongside revenue "
                        "to understand cash conversion dynamics."
                    ),
                },
                {
                    "sub_domain": "liquidity",
                    "action": (
                        "Monitor receivables variability to assess "
                        "collection pattern consistency."
                    ),
                },
                {
                    "sub_domain": "capital_structure",
                    "action": (
                        "Assess debt levels relative to assets "
                        "to understand capital composition."
                    ),
                },
                {
                    "sub_domain": "capital_structure",
                    "action": (
                        "Observe how debt and asset signals evolve "
                        "together over time."
                    ),
                },
                {
                    "sub_domain": "capital_structure",
                    "action": (
                        "Use balance sheet relationships as context "
                        "for funding discussions."
                    ),
                },
                {
                    "sub_domain": "liquidity",
                    "action": (
                        "Review liquidity proxy continuity "
                        "to support cash flow interpretation."
                    ),
                },
                {
                    "sub_domain": "capital_structure",
                    "action": (
                        "Contextualize short- and long-term obligations "
                        "within the broader financial structure."
                    ),
                },
            ])
    
        # =================================================
        # BANKING HEALTH (OPTIONAL)
        # =================================================
        if "loans_mean" in kpis:
            recommendations.extend([
                {
                    "sub_domain": "banking",
                    "action": (
                        "Review loan portfolio size and distribution "
                        "to understand credit exposure scale."
                    ),
                },
                {
                    "sub_domain": "banking",
                    "action": (
                        "Monitor loan variability to assess "
                        "portfolio dispersion."
                    ),
                },
                {
                    "sub_domain": "banking",
                    "action": (
                        "Observe non-performing assets alongside "
                        "total loans for composition context."
                    ),
                },
                {
                    "sub_domain": "banking",
                    "action": (
                        "Review collateral signals to contextualize "
                        "secured lending exposure."
                    ),
                },
                {
                    "sub_domain": "banking",
                    "action": (
                        "Track loan and collateral dynamics "
                        "over time."
                    ),
                },
                {
                    "sub_domain": "banking",
                    "action": (
                        "Use banking signal continuity "
                        "to support portfolio interpretation."
                    ),
                },
                {
                    "sub_domain": "banking",
                    "action": (
                        "Interpret banking metrics within "
                        "overall financial structure."
                    ),
                },
            ])
    
        # =================================================
        # INTEGRITY / DATA QUALITY
        # =================================================
        if "expense_benford_deviation" in kpis:
            recommendations.extend([
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Review expense recording practices "
                        "to understand value distribution patterns."
                    ),
                },
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Monitor diversity of expense values "
                        "to assess recording consistency."
                    ),
                },
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Observe zero or negative expense entries "
                        "for contextual interpretation."
                    ),
                },
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Compare expense distribution shapes "
                        "across periods."
                    ),
                },
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Use digit-distribution patterns "
                        "as contextual integrity signals."
                    ),
                },
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Review expense aggregation logic "
                        "for consistency."
                    ),
                },
                {
                    "sub_domain": "integrity",
                    "action": (
                        "Interpret anomaly signals alongside "
                        "other financial indicators."
                    ),
                },
            ])
    
        # =================================================
        # GUARANTEED FALLBACK
        # =================================================
        if not recommendations:
            recommendations.append({
                "sub_domain": "mixed",
                "action": (
                    "Continue monitoring available financial signals "
                    "to expand analytical depth as data coverage improves."
                ),
            })
    
        return recommendations

# =====================================================
# Finance Domain — Block 7
# Domain Detector (Boundary-Safe, Enterprise)
# =====================================================
class FinanceDomainDetector(BaseDomainDetector):
    """
    Boundary-safe detector for the Finance domain.

    Detects:
    - Corporate finance (P&L, Balance Sheet)
    - Banking & credit datasets
    - Financial statements & ledgers

    Prevents collision with:
    - Retail / Ecommerce transactions
    - Marketing performance data
    - Generic event logs
    """

    domain_name = "finance"

    # -----------------------------
    # Semantic Token Clusters
    # -----------------------------
    PNL_TOKENS = {
        "revenue", "sales", "income",
        "expense", "cost", "cogs", "opex",
        "profit", "ebit", "ebitda",
    }

    BALANCE_SHEET_TOKENS = {
        "asset", "assets",
        "liability", "liabilities",
        "equity", "debt",
    }

    FINANCE_OPS_TOKENS = {
        "loan", "interest",
        "receivable", "payable",
        "ledger", "npa",
    }

    GENERIC_TOKENS = {
        "price", "volume", "date", "id"
    }

    # -----------------------------
    # Detection Logic
    # -----------------------------
    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        # Lowercase → original column mapping
        col_map = {str(c).lower(): c for c in df.columns}

        def tokenize(col: str) -> set:
            return set(col.replace("_", " ").split())

        tokenized_cols = {
            lc: tokenize(lc)
            for lc in col_map.keys()
        }

        # -----------------------------
        # Semantic Hits
        # -----------------------------
        pnl_hits = {
            col_map[c]
            for c, toks in tokenized_cols.items()
            if toks & self.PNL_TOKENS
        }

        bs_hits = {
            col_map[c]
            for c, toks in tokenized_cols.items()
            if toks & self.BALANCE_SHEET_TOKENS
        }

        ops_hits = {
            col_map[c]
            for c, toks in tokenized_cols.items()
            if toks & self.FINANCE_OPS_TOKENS
        }

        generic_hits = {
            col_map[c]
            for c, toks in tokenized_cols.items()
            if toks & self.GENERIC_TOKENS
        }

        # -----------------------------
        # Numeric Validation (CRITICAL)
        # -----------------------------
        numeric_finance_cols = {
            c for c in (pnl_hits | bs_hits | ops_hits)
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        }

        # -----------------------------
        # Confidence Construction
        # -----------------------------
        signal_groups_present = sum([
            bool(pnl_hits),
            bool(bs_hits),
            bool(ops_hits),
        ])

        if not numeric_finance_cols or signal_groups_present == 0:
            return DomainDetectionResult(
                None,
                0.0,
                {
                    "reason": "No numeric finance signals detected",
                    "pnl_columns": sorted(pnl_hits),
                    "balance_sheet_columns": sorted(bs_hits),
                    "finance_ops_columns": sorted(ops_hits),
                },
            )

        confidence = signal_groups_present / 3.0

        # Penalize generic-only dominance
        if signal_groups_present == 1 and generic_hits:
            confidence *= 0.5

        # Strong GL + P&L signature boost
        has_gl = any(
            "gl" in str(c).lower() and "account" in str(c).lower()
            for c in df.columns
        )

        if has_gl and pnl_hits:
            confidence = max(confidence, 0.85)

        confidence = round(min(confidence, 0.95), 2)

        return DomainDetectionResult(
            domain="finance",
            confidence=confidence,
            signals={
                "pnl_columns": sorted(pnl_hits),
                "balance_sheet_columns": sorted(bs_hits),
                "finance_ops_columns": sorted(ops_hits),
                "numeric_finance_columns": sorted(numeric_finance_cols),
                "generic_columns": sorted(generic_hits),
                "has_gl_account": has_gl,
            },
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
