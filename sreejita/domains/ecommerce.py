import pandas as pd
import numpy as np
import matplotlib

# Headless backend for CI / server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# HELPERS — ECOMMERCE (DOMAIN-AGNOSTIC, SAFE)
# =====================================================

def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    """
    Safe division helper.

    Guarantees:
    - Never raises
    - Never invents values
    - Returns None on invalid input
    """
    try:
        if d in (0, None) or pd.isna(d):
            return None
        if n in (None,) or pd.isna(n):
            return None
        return float(n) / float(d)
    except Exception:
        return None


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Ecommerce-oriented time column detector.

    Supports:
    - session timelines
    - visit timelines
    - order timelines

    Notes:
    - Used strictly as a fallback
    - Semantic resolution is always preferred
    """

    if df is None or df.empty:
        return None

    candidates = [
        "session_date",
        "visit_date",
        "timestamp",
        "order_date",
        "created_at",
        "date",
    ]

    for col in df.columns:
        col_l = str(col).lower()
        if any(k in col_l for k in candidates):
            try:
                sample = df[col].dropna().iloc[:5]
                if sample.empty:
                    continue
                pd.to_datetime(sample, errors="raise")
                return col
            except Exception:
                continue

    return None


def _detect_session_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detect session / traffic related columns.

    Guarantees:
    - No assumptions
    - Safe fallbacks
    - No mutation
    """
    return {
        "sessions": resolve_column(df, "sessions") or resolve_column(df, "visits"),
        "users": resolve_column(df, "users") or resolve_column(df, "visitors"),
    }


def _detect_funnel_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detect funnel-related columns safely.

    IMPORTANT:
    - Detection only
    - No metric computation
    """
    return {
        "orders": resolve_column(df, "orders") or resolve_column(df, "transactions"),
        "add_to_cart": resolve_column(df, "add_to_cart") or resolve_column(df, "atc"),
        "checkout": resolve_column(df, "checkout") or resolve_column(df, "begin_checkout"),
    }


def _compute_conversion_proxy(
    df: pd.DataFrame,
    session_col: Optional[str],
    order_col: Optional[str],
) -> Optional[float]:
    """
    Conversion proxy (orders / sessions).

    IMPORTANT:
    - INTERNAL helper only
    - NOT a KPI
    - Must never be exposed directly
    """

    if (
        df is None
        or df.empty
        or not session_col
        or not order_col
        or session_col not in df.columns
        or order_col not in df.columns
    ):
        return None

    sessions = pd.to_numeric(df[session_col], errors="coerce").sum()
    orders = pd.to_numeric(df[order_col], errors="coerce").sum()

    return _safe_div(orders, sessions)


def _compute_return_rate_proxy(
    df: pd.DataFrame,
    returned_col: Optional[str],
    order_col: Optional[str],
) -> Optional[float]:
    """
    Return / cancellation proxy.

    IMPORTANT:
    - INTERNAL helper only
    - NOT a KPI
    - Safe under weak or partial data
    """

    if (
        df is None
        or df.empty
        or not returned_col
        or not order_col
        or returned_col not in df.columns
        or order_col not in df.columns
    ):
        return None

    returned = pd.to_numeric(df[returned_col], errors="coerce").sum()
    orders = pd.to_numeric(df[order_col], errors="coerce").sum()

    return _safe_div(returned, orders)


class EcommerceDomain(BaseDomain):
    name = "ecommerce"
    description = "Universal E-Commerce Analytics (Traffic, Conversion, Funnel, Retention)"

    # -------------------------------------------------
    # PREPROCESS (SEMANTIC, SAFE, NO KPI LOGIC)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ecommerce preprocess:
        - Semantic column resolution (authoritative)
        - Numeric and datetime normalization
        - Sub-domain routing preparation only
        - NO KPI logic
        - NO confidence logic
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("EcommerceDomain.preprocess expects a DataFrame")

        # Defensive copy
        df = df.copy(deep=False)

        # -------------------------------------------------
        # TIME COLUMN (SEMANTIC FIRST, FALLBACK SAFE)
        # -------------------------------------------------
        self.time_col = _detect_time_column(df)

        # -------------------------------------------------
        # CANONICAL SEMANTIC COLUMN RESOLUTION (ONCE)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # -------- TRAFFIC --------
            "sessions": resolve_column(df, "sessions") or resolve_column(df, "visits"),
            "users": resolve_column(df, "users") or resolve_column(df, "visitors"),
            "pageviews": resolve_column(df, "pageviews") or resolve_column(df, "screen_views"),
            "bounce": resolve_column(df, "bounce_rate"),

            # -------- FUNNEL --------
            "add_to_cart": resolve_column(df, "add_to_cart") or resolve_column(df, "atc"),
            "checkout": resolve_column(df, "checkout") or resolve_column(df, "begin_checkout"),
            "orders": resolve_column(df, "orders") or resolve_column(df, "transactions"),
            "revenue": (
                resolve_column(df, "revenue")
                or resolve_column(df, "total_revenue")
                or resolve_column(df, "sales")
            ),
            "returns": resolve_column(df, "returns") or resolve_column(df, "refunds"),

            # -------- CUSTOMER --------
            "customer": resolve_column(df, "customer_id") or resolve_column(df, "user_id"),

            # -------- DIMENSIONS --------
            "source": resolve_column(df, "source") or resolve_column(df, "channel"),
            "device": resolve_column(df, "device") or resolve_column(df, "platform"),
            "product": resolve_column(df, "product_name") or resolve_column(df, "sku"),
            "category": resolve_column(df, "category"),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        numeric_keys = {
            "sessions",
            "users",
            "pageviews",
            "add_to_cart",
            "checkout",
            "orders",
            "revenue",
            "returns",
        }

        for key in numeric_keys:
            col = self.cols.get(key)
            if col and col in df.columns:
                if df[col].dtype == object:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"[^\d\.\-]", "", regex=True)
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Bounce rate normalization (0–1)
        bounce_col = self.cols.get("bounce")
        if bounce_col and bounce_col in df.columns:
            if df[bounce_col].dtype == object:
                df[bounce_col] = df[bounce_col].astype(str).str.replace("%", "", regex=False)
            df[bounce_col] = pd.to_numeric(df[bounce_col], errors="coerce")
            if df[bounce_col].dropna().median() and df[bounce_col].dropna().median() > 1:
                df[bounce_col] = df[bounce_col] / 100.0
            df[bounce_col] = df[bounce_col].clip(0, 1)

        # -------------------------------------------------
        # DATETIME NORMALIZATION
        # -------------------------------------------------
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # -------------------------------------------------
    # SAFE RUN WRAPPER (PHASE-5 OUTPUT CONTRACT)
    # -------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        *,
        visual_output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute the ecommerce pipeline with Phase-5 safety guarantees.

        Guarantees:
        - Deterministic behavior
        - No KPI invention
        - No confidence invention
        - No insights or recommendations under weak data
        - Stable output shape
        """

        result: Dict[str, Any] = {
            "kpis": {},
            "insights": [],
            "recommendations": [],
            "visuals": [],
        }

        try:
            if not self.validate_data(df):
                return result

            df = self.preprocess(df)
            if df is None or df.empty:
                return result

            # Sub-domain inference only (no KPI exposure)
            self.calculate_kpis(df)

            # Visuals are allowed, descriptive only
            if visual_output_dir is not None:
                try:
                    visuals = self.generate_visuals(df, visual_output_dir)
                    result["visuals"] = visuals if isinstance(visuals, list) else []
                except Exception:
                    result["visuals"] = []

            return result

        except Exception:
            return result


    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ecommerce KPI stub (Phase 5 – trust-correction).

        Guarantees:
        - NO KPI invention
        - NO inline KPI math exposed
        - NO confidence creation or mutation
        - NO capability tagging
        - Deterministic behavior
        - Safe under weak or partial data

        NOTE:
        - Ecommerce KPIs are not yet registry-approved
        - This method intentionally returns an EMPTY dict
        """

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            self._active_subdomains = {}
            return {}

        # -------------------------------------------------
        # SUB-DOMAIN ROUTING (INTERNAL ONLY)
        # -------------------------------------------------
        # For Phase 5, we infer *presence* only, not metrics.
        active_subs: Dict[str, float] = {}

        # Traffic
        if self.cols.get("sessions") or self.cols.get("users"):
            active_subs["traffic"] = 0.6

        # Funnel / Conversion
        if self.cols.get("orders") or self.cols.get("add_to_cart"):
            active_subs["conversion"] = 0.6

        # Revenue
        if self.cols.get("revenue"):
            active_subs["revenue"] = 0.6

        # Customer / Retention
        if self.cols.get("customer"):
            active_subs["customer"] = 0.6

        # Operations
        if self.cols.get("returns"):
            active_subs["operations"] = 0.6

        # Store for visual routing only
        self._active_subdomains = active_subs

        # -------------------------------------------------
        # PHASE 5 RULE:
        # Do NOT expose ecommerce KPIs yet
        # -------------------------------------------------
        return {}


    # ---------------- VISUALS (8 CANDIDATES) ----------------
    
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Ecommerce Visual Engine (Phase 5 – descriptive only)

        Guarantees:
        - Visuals are purely descriptive
        - No KPI dependency
        - No confidence fabrication
        - Graceful degradation on weak data
        - Stable visual object shape
        """

        visuals: List[Dict[str, Any]] = []

        if df is None or df.empty:
            return visuals

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        c = self.cols

        # -------------------------------------------------
        # ACTIVE SUB-DOMAINS (ROUTING ONLY)
        # -------------------------------------------------
        active_subs = getattr(self, "_active_subdomains", {}) or {}

        # -------------------------------------------------
        # HELPERS
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
                "confidence": None,   # Phase 5: never invent confidence
            })

        def human_fmt(x, _):
            if abs(x) >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))

        # =================================================
        # TRAFFIC & ACQUISITION
        # =================================================
        if "traffic" in active_subs:
            if self.time_col and c.get("sessions"):
                fig, ax = plt.subplots(figsize=(7, 4))
                (
                    df.set_index(self.time_col)
                    .resample("M")[c["sessions"]]
                    .sum()
                    .plot(ax=ax)
                )
                ax.set_title("Traffic Trend (Sessions)")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "traffic_trend.png",
                    "Visitor volume over time",
                    0.9,
                    "traffic",
                    "traffic",
                    "time",
                )

            if c.get("source") and c.get("sessions"):
                fig, ax = plt.subplots(figsize=(7, 4))
                (
                    df.groupby(c["source"])[c["sessions"]]
                    .sum()
                    .nlargest(7)
                    .sort_values()
                    .plot.barh(ax=ax)
                )
                ax.set_title("Top Traffic Sources")
                save(
                    fig,
                    "traffic_sources.png",
                    "Acquisition mix by source",
                    0.8,
                    "traffic",
                    "traffic",
                    "entity",
                )

            if c.get("device") and c.get("sessions"):
                fig, ax = plt.subplots(figsize=(6, 4))
                (
                    df.groupby(c["device"])[c["sessions"]]
                    .sum()
                    .plot.pie(ax=ax, autopct="%1.1f%%")
                )
                ax.set_ylabel("")
                ax.set_title("Traffic by Device")
                save(
                    fig,
                    "device_mix.png",
                    "Platform mix",
                    0.7,
                    "traffic",
                    "traffic",
                    "composition",
                )

        # =================================================
        # CONVERSION & FUNNEL
        # =================================================
        if "conversion" in active_subs:
            if c.get("sessions") and c.get("add_to_cart") and c.get("orders"):
                fig, ax = plt.subplots(figsize=(7, 4))
                funnel = [
                    df[c["sessions"]].sum(),
                    df[c["add_to_cart"]].sum(),
                    df[c["orders"]].sum(),
                ]
                ax.bar(["Sessions", "Add to Cart", "Orders"], funnel)
                ax.set_title("Conversion Funnel")
                save(
                    fig,
                    "conversion_funnel.png",
                    "High-level funnel progression",
                    0.95,
                    "conversion",
                    "funnel",
                    "stage",
                )

            if self.time_col and c.get("orders"):
                fig, ax = plt.subplots(figsize=(7, 4))
                (
                    df.set_index(self.time_col)
                    .resample("M")[c["orders"]]
                    .sum()
                    .plot(ax=ax)
                )
                ax.set_title("Orders Over Time")
                save(
                    fig,
                    "orders_trend.png",
                    "Order volume trend",
                    0.85,
                    "conversion",
                    "orders",
                    "time",
                )

        # =================================================
        # REVENUE & ECONOMICS
        # =================================================
        if "revenue" in active_subs:
            if self.time_col and c.get("revenue"):
                fig, ax = plt.subplots(figsize=(7, 4))
                (
                    df.set_index(self.time_col)
                    .resample("M")[c["revenue"]]
                    .sum()
                    .plot(ax=ax)
                )
                ax.set_title("Revenue Trend")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "revenue_trend.png",
                    "Revenue over time",
                    0.9,
                    "revenue",
                    "revenue",
                    "time",
                )

            if c.get("revenue") and c.get("orders"):
                mask = df[c["orders"]] > 0
                if mask.sum() >= 10:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    (
                        (df.loc[mask, c["revenue"]] / df.loc[mask, c["orders"]])
                        .hist(ax=ax, bins=20)
                    )
                    ax.set_title("Order Value Distribution")
                    save(
                        fig,
                        "order_value_dist.png",
                        "Distribution of order values",
                        0.8,
                        "revenue",
                        "distribution",
                        "value",
                    )

        # =================================================
        # CUSTOMER & RETENTION
        # =================================================
        if "customer" in active_subs:
            if c.get("customer"):
                freq = df[c["customer"]].value_counts()
                if not freq.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    freq.clip(upper=5).value_counts().sort_index().plot.bar(ax=ax)
                    ax.set_title("Orders per Customer")
                    save(
                        fig,
                        "orders_per_customer.png",
                        "Repeat purchase behavior",
                        0.7,
                        "customer",
                        "customer",
                        "distribution",
                    )

        # =================================================
        # OPERATIONAL FRICTION
        # =================================================
        if "operations" in active_subs:
            if c.get("returns"):
                fig, ax = plt.subplots(figsize=(6, 4))
                df[c["returns"]].dropna().hist(ax=ax, bins=20)
                ax.set_title("Returns Distribution")
                save(
                    fig,
                    "returns_distribution.png",
                    "Observed return volume distribution",
                    0.8,
                    "operations",
                    "returns",
                    "distribution",
                )

        # -------------------------------------------------
        # FINAL SORT — MAX 6 (CONSISTENT WITH RETAIL)
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:6]


    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Ecommerce Insight Engine (Phase 5 – silence by design)

        Guarantees:
        - No KPI-based inference
        - No narrative generation
        - No confidence fabrication
        - No filler insights
        - Deterministic output

        Phase 5 rule:
        If KPIs are not approved, insights MUST be suppressed.
        """

        return []


    def generate_composite_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Composite insights are intentionally disabled in Phase 5.

        Reason:
        - Composite insights require trusted KPIs
        - Ecommerce KPIs are not registry-approved yet
        """

        return []
    

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ecommerce Recommendation Engine (Phase 5 – disabled)

        Guarantees:
        - No generic advice
        - No KPI-dependent recommendations
        - No confidence fabrication
        - Deterministic silence

        Phase 5 rule:
        Recommendations require trusted KPIs and insights.
        Ecommerce does not expose either yet.
        """

        return []


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class EcommerceDomainDetector(BaseDomainDetector):
    domain_name = "ecommerce"

    TOKENS = {
        "session",
        "sessions",
        "cart",
        "add_to_cart",
        "checkout",
        "order",
        "orders",
        "transaction",
        "revenue",
        "refund",
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        cols = {c.lower() for c in df.columns}

        # -------------------------------------------------
        # 🚫 FINANCE LEDGER NEGATIVE GATE (CRITICAL FIX)
        # -------------------------------------------------
        has_gl_account = any("gl" in c and "account" in c for c in cols)
        has_profit = any("profit" in c for c in cols)
        has_expense = any("expense" in c or "cost" in c for c in cols)

        # Ledger-style dataset → not ecommerce
        if has_gl_account and has_profit and has_expense:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={"reason": "ledger_dataset"},
            )

        # -------------------------------------------------
        # TOKEN MATCHING (UNCHANGED)
        # -------------------------------------------------
        hits = [
            c for c in cols
            if any(t in c for t in self.TOKENS)
        ]

        confidence = min(len(hits) / 4, 1.0)

        has_sessions = any("session" in c for c in cols)
        has_cart = any("cart" in c for c in cols)
        has_checkout = any("checkout" in c for c in cols)
        has_orders = any("order" in c or "transaction" in c for c in cols)
        has_revenue = any("revenue" in c or "sales" in c for c in cols)

        if has_sessions and (has_cart or has_checkout):
            confidence = max(confidence, 0.85)

        if has_orders and has_revenue:
            confidence = max(confidence, 0.90)

        if has_sessions and has_orders and has_revenue:
            confidence = max(confidence, 0.95)

        return DomainDetectionResult(
            "ecommerce",
            round(confidence, 2),
            {"matched_columns": sorted(hits)},
        )


def register(registry):
    registry.register(
        "ecommerce",
        EcommerceDomain,
        EcommerceDomainDetector
    )

