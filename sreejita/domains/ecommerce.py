import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# HELPERS (ECOMMERCE-SAFE)
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Ecommerce-specific time detector.
    Supports session, visit, and order timelines.
    """
    candidates = [
        "session_date",
        "visit_date",
        "timestamp",
        "order_date",
        "created_at",
        "date",
    ]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except Exception:
                continue
    return None


def _detect_session_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detects session / traffic related columns.
    """
    return {
        "sessions": resolve_column(df, "sessions") or resolve_column(df, "visits"),
        "users": resolve_column(df, "users") or resolve_column(df, "visitors"),
    }


def _detect_funnel_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detects funnel-related columns safely.
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
    Safe conversion proxy: orders / sessions.
    """
    if (
        not session_col
        or not order_col
        or session_col not in df.columns
        or order_col not in df.columns
    ):
        return None

    sessions = df[session_col].sum()
    orders = df[order_col].sum()

    return _safe_div(orders, sessions)


def _compute_return_rate_proxy(
    df: pd.DataFrame,
    returned_col: Optional[str],
    order_col: Optional[str],
) -> Optional[float]:
    """
    Safe return / cancellation proxy.
    """
    if (
        not returned_col
        or not order_col
        or returned_col not in df.columns
        or order_col not in df.columns
    ):
        return None

    returned = df[returned_col].sum()
    orders = df[order_col].sum()

    return _safe_div(returned, orders)


class EcommerceDomain(BaseDomain):
    name = "ecommerce"
    description = "Universal E-Commerce Analytics (Traffic, Conversion, Funnel, Retention)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
    
        # -------------------------------------------------
        # 1. Resolve columns ONCE (Ecommerce-specific)
        # -------------------------------------------------
        self.cols = {
            # Traffic
            "sessions": resolve_column(df, "sessions") or resolve_column(df, "visits"),
            "users": resolve_column(df, "users") or resolve_column(df, "visitors"),
            "pageviews": resolve_column(df, "pageviews") or resolve_column(df, "screen_views"),
            "bounce": resolve_column(df, "bounce_rate"),
    
            # Funnel
            "add_to_cart": resolve_column(df, "add_to_cart") or resolve_column(df, "atc"),
            "checkout": resolve_column(df, "checkout") or resolve_column(df, "begin_checkout"),
            "orders": resolve_column(df, "orders") or resolve_column(df, "transactions"),
            "revenue": (
                resolve_column(df, "revenue")
                or resolve_column(df, "total_revenue")
                or resolve_column(df, "sales")
            ),
            "returns": resolve_column(df, "returns") or resolve_column(df, "refunds"),
    
            # Retention / identity (light touch)
            "customer": resolve_column(df, "customer_id") or resolve_column(df, "user_id"),
    
            # Dimensions
            "source": resolve_column(df, "source") or resolve_column(df, "channel"),
            "device": resolve_column(df, "device") or resolve_column(df, "platform"),
            "product": resolve_column(df, "product_name") or resolve_column(df, "sku"),
            "category": resolve_column(df, "category"),
        }
    
        # -------------------------------------------------
        # 2. Store detected funnel & session columns (NEW)
        # -------------------------------------------------
        self.session_cols = _detect_session_columns(df)
        self.funnel_cols = _detect_funnel_columns(df)
    
        # -------------------------------------------------
        # 3. Define Ecommerce sub-domains (NEW)
        # -------------------------------------------------
        self.sub_domains = {
            "traffic": "Traffic & Acquisition",
            "conversion": "Conversion & Funnel",
            "revenue": "Revenue & Economics",
            "customer": "Customer & Retention Signals",
            "operations": "Operational Friction",
        }
    
        # KPI → sub-domain mapping (used later)
        self._domain_kpi_map = {
            "traffic": [
                "total_sessions",
                "total_users",
                "avg_bounce_rate",
            ],
            "conversion": [
                "conversion_rate",
                "conversion_rate_proxy",
                "cart_abandonment_rate",
                "checkout_dropoff_rate",
            ],
            "revenue": [
                "aov",
                "revenue_per_session",
                "revenue_per_user",
            ],
            "customer": [
                "repeat_user_rate",
                "orders_per_customer",
            ],
            "operations": [
                "return_rate",
            ],
        }
    
        # -------------------------------------------------
        # 4. Numeric safety & normalization
        # -------------------------------------------------
        numeric_cols = [
            "sessions",
            "users",
            "pageviews",
            "add_to_cart",
            "checkout",
            "orders",
            "revenue",
            "returns",
        ]
    
        for k in numeric_cols:
            col = self.cols.get(k)
            if col and col in df.columns:
                if df[col].dtype == object:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"[,$]", "", regex=True)
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
        # Bounce rate normalization
        if self.cols.get("bounce") and self.cols["bounce"] in df.columns:
            b = self.cols["bounce"]
            if df[b].dtype == object:
                df[b] = df[b].astype(str).str.replace("%", "", regex=False)
            df[b] = pd.to_numeric(df[b], errors="coerce")
            if df[b].dropna().median() > 1:
                df[b] = df[b] / 100.0
            df[b] = df[b].clip(0, 1)
    
        # -------------------------------------------------
        # 5. Date cleaning
        # -------------------------------------------------
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)
    
        # -------------------------------------------------
        # 6. Data completeness (optional but useful)
        # -------------------------------------------------
        present = sum(1 for v in self.cols.values() if v)
        self.data_completeness = round(present / max(len(self.cols), 1), 2)

        # -------------------------------------------------
        # 7. Dataset-shape-aware sub-domain suppression
        # -------------------------------------------------
        missing_funnel_signals = not any([
            self.cols.get("sessions"),
            self.cols.get("add_to_cart"),
            self.cols.get("checkout"),
        ])
        
        if missing_funnel_signals:
            # Suppress funnel-heavy sub-domains
            self.sub_domains.pop("traffic", None)
            self.sub_domains.pop("conversion", None)
        
            # Narrow KPI map accordingly
            self._domain_kpi_map = {
                k: v for k, v in self._domain_kpi_map.items()
                if k in ("revenue", "customer", "operations")
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
        Execute the ecommerce pipeline with stable, safe outputs.

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
        kpis: Dict[str, Any] = {}
        c = self.cols
    
        # -------------------------------------------------
        # Governance metadata
        # -------------------------------------------------
        kpis["_confidence"] = {}
        kpis["_kpi_capabilities"] = {}
        kpis["_domain_kpi_map"] = self._domain_kpi_map
        kpis["sub_domains"] = self.sub_domains
        kpis["data_completeness"] = getattr(self, "data_completeness", 0.7)
    
        def record(key, value, capability, confidence=0.75):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return
            kpis[key] = value
            kpis["_confidence"][key] = round(confidence, 2)
            kpis["_kpi_capabilities"][key] = capability
    
        # =================================================
        # SUB-DOMAIN: TRAFFIC & ACQUISITION
        # =================================================
        if c.get("sessions"):
            record("total_sessions", df[c["sessions"]].sum(), "volume", 0.90)
    
        if c.get("users"):
            record("total_users", df[c["users"]].sum(), "volume", 0.85)
    
        if c.get("pageviews"):
            record("total_pageviews", df[c["pageviews"]].sum(), "volume", 0.80)
    
        if c.get("sessions") and c.get("users"):
            record(
                "sessions_per_user",
                _safe_div(df[c["sessions"]].sum(), df[c["users"]].sum()),
                "quality",
                0.80,
            )
    
        if c.get("bounce"):
            record("avg_bounce_rate", df[c["bounce"]].mean(), "quality", 0.75)
    
        # =================================================
        # SUB-DOMAIN: CONVERSION & FUNNEL
        # =================================================
        orders_sum = df[c["orders"]].sum() if c.get("orders") else None
    
        if c.get("orders") and c.get("sessions"):
            record(
                "conversion_rate",
                min(max(_safe_div(orders_sum, df[c["sessions"]].sum()), 0.0), 1.0),
                "quality",
                0.85,
            )
    
        elif c.get("orders") and c.get("users"):
            record(
                "conversion_rate_proxy",
                min(max(_safe_div(orders_sum, df[c["users"]].sum()), 0.0), 1.0),
                "quality",
                0.70,
            )
    
        if c.get("add_to_cart") and c.get("orders"):
            atc = df[c["add_to_cart"]].sum()
            if atc > 0:
                record(
                    "cart_abandonment_rate",
                    min(max(1.0 - (orders_sum / atc), 0.0), 1.0),
                    "variance",
                    0.75,
                )
    
        if c.get("checkout") and c.get("orders"):
            chk = df[c["checkout"]].sum()
            if chk > 0:
                record(
                    "checkout_dropoff_rate",
                    min(max(1.0 - (orders_sum / chk), 0.0), 1.0),
                    "variance",
                    0.75,
                )
    
        if c.get("orders") and c.get("users"):
            record(
                "orders_per_user",
                _safe_div(orders_sum, df[c["users"]].sum()),
                "volume",
                0.70,
            )
    
        # =================================================
        # SUB-DOMAIN: REVENUE & ECONOMICS
        # =================================================
        if c.get("revenue"):
            record("total_revenue", df[c["revenue"]].sum(), "cost", 0.90)
    
        if c.get("revenue") and c.get("orders"):
            record(
                "aov",
                _safe_div(df[c["revenue"]].sum(), orders_sum),
                "cost",
                0.85,
            )
    
        if c.get("revenue") and c.get("sessions"):
            record(
                "revenue_per_session",
                _safe_div(df[c["revenue"]].sum(), df[c["sessions"]].sum()),
                "cost",
                0.75,
            )
    
        if c.get("revenue") and c.get("users"):
            record(
                "revenue_per_user",
                _safe_div(df[c["revenue"]].sum(), df[c["users"]].sum()),
                "cost",
                0.75,
            )
    
        # =================================================
        # SUB-DOMAIN: CUSTOMER & RETENTION (PROXY)
        # =================================================
        if c.get("customer") and c.get("orders"):
            orders_per_customer = (
                df.groupby(c["customer"])[c["orders"]].sum().mean()
            )
            record(
                "orders_per_customer",
                orders_per_customer,
                "volume",
                0.70,
            )
    
        if c.get("customer"):
            freq = df[c["customer"]].value_counts()
            repeat_rate = _safe_div((freq > 1).sum(), len(freq))
            record(
                "repeat_user_rate",
                repeat_rate,
                "quality",
                0.65,
            )
    
        # =================================================
        # SUB-DOMAIN: OPERATIONAL FRICTION
        # =================================================
        if c.get("returns") and c.get("orders"):
            record(
                "return_rate",
                min(
                    max(
                        _safe_div(df[c["returns"]].sum(), orders_sum),
                        0.0,
                    ),
                    1.0,
                ),
                "variance",
                0.70,
            )
    
        return kpis

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        kpis = self.calculate_kpis(df)
    
        def save(fig, name, caption, importance, sub_domain):
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": importance,
                "confidence": 0.8,
                "sub_domain": sub_domain,
            })
    
        def human_fmt(x, _):
            if abs(x) >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000:
                return f"{x/1_000:.0f}K"
            return str(int(x))
    
        # =================================================
        # SUB-DOMAIN: TRAFFIC & ACQUISITION (≥3)
        # =================================================
        if self.time_col and c.get("sessions"):
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(self.time_col).resample("M")[c["sessions"]].sum().plot(ax=ax)
            ax.set_title("Traffic Trend (Sessions)")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "traffic_trend.png", "Visitor volume trend", 0.9, "traffic")
    
        if c.get("source") and c.get("sessions"):
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(c["source"])[c["sessions"]].sum().nlargest(7).sort_values().plot.barh(ax=ax)
            ax.set_title("Top Traffic Sources")
            save(fig, "traffic_sources.png", "Acquisition mix", 0.8, "traffic")
    
        if c.get("device") and c.get("sessions"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["device"])[c["sessions"]].sum().plot.pie(ax=ax, autopct="%1.1f%%")
            ax.axis("equal")
            ax.set_ylabel("")
            ax.set_title("Traffic by Device")
            save(fig, "device_mix.png", "Platform mix", 0.7, "traffic")
    
        # =================================================
        # SUB-DOMAIN: CONVERSION & FUNNEL (≥3)
        # =================================================
        if c.get("sessions") and c.get("add_to_cart") and c.get("orders"):
            fig, ax = plt.subplots(figsize=(7, 4))
            funnel = [
                df[c["sessions"]].sum(),
                df[c["add_to_cart"]].sum(),
                df[c["orders"]].sum(),
            ]
            ax.bar(["Sessions", "Add to Cart", "Orders"], funnel)
            ax.set_title("Conversion Funnel")
            save(fig, "conversion_funnel.png", "Funnel drop-off", 0.95, "conversion")
    
        if self.time_col and c.get("orders"):
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(self.time_col).resample("M")[c["orders"]].sum().plot(ax=ax)
            ax.set_title("Orders Trend")
            save(fig, "orders_trend.png", "Order momentum", 0.85, "conversion")
    
        if c.get("sessions") and c.get("orders"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                df[c["sessions"]],
                df[c["orders"]],
                alpha=0.4,
            )
            ax.set_xlabel("Sessions")
            ax.set_ylabel("Orders")
            ax.set_title("Sessions vs Orders")
            save(fig, "sessions_vs_orders.png", "Traffic efficiency", 0.8, "conversion")
    
        # =================================================
        # SUB-DOMAIN: REVENUE & ECONOMICS (≥3)
        # =================================================
        if self.time_col and c.get("revenue"):
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(self.time_col).resample("M")[c["revenue"]].sum().plot(ax=ax)
            ax.set_title("Revenue Trend")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "revenue_trend.png", "Revenue momentum", 0.9, "revenue")
    
        if c.get("revenue") and c.get("orders"):
            mask = df[c["orders"]] > 0
            if mask.sum() > 10:
                fig, ax = plt.subplots(figsize=(6, 4))
                (df.loc[mask, c["revenue"]] / df.loc[mask, c["orders"]]).hist(ax=ax, bins=20)
                ax.set_title("Order Value Distribution")
                save(fig, "aov_distribution.png", "Spending distribution", 0.8, "revenue")
    
        if c.get("revenue") and c.get("sessions"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                df[c["sessions"]],
                df[c["revenue"]],
                alpha=0.4,
            )
            ax.set_xlabel("Sessions")
            ax.set_ylabel("Revenue")
            ax.set_title("Revenue vs Sessions")
            save(fig, "revenue_vs_sessions.png", "Monetization efficiency", 0.75, "revenue")
    
        # =================================================
        # SUB-DOMAIN: CUSTOMER & RETENTION (≥2)
        # =================================================
        if c.get("customer"):
            freq = df[c["customer"]].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            freq.clip(upper=5).value_counts().sort_index().plot.bar(ax=ax)
            ax.set_title("Orders per Customer")
            save(fig, "orders_per_customer.png", "Repeat behavior", 0.7, "customer")
    
        # =================================================
        # SUB-DOMAIN: OPERATIONAL FRICTION (≥2)
        # =================================================
        if c.get("returns"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["returns"]].hist(ax=ax, bins=20)
            ax.set_title("Returns Distribution")
            save(fig, "returns_dist.png", "Return behavior", 0.8, "operations")
    
        if c.get("returns") and c.get("orders"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df[c["orders"]], df[c["returns"]], alpha=0.4)
            ax.set_xlabel("Orders")
            ax.set_ylabel("Returns")
            ax.set_title("Orders vs Returns")
            save(fig, "orders_vs_returns.png", "Operational leakage", 0.75, "operations")
    
        return visuals

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []
    
        # =================================================
        # COMPOSITE INSIGHTS (PRIMARY INTELLIGENCE)
        # =================================================
        insights.extend(self.generate_composite_insights(df, kpis))
    
        # Titles already used (avoid duplication)
        used_titles = {i["title"] for i in insights if isinstance(i, dict)}
    
        # =================================================
        # ATOMIC FALLBACKS (SAFETY NET)
        # =================================================
        cr = kpis.get("conversion_rate")
        bounce = kpis.get("avg_bounce_rate")
        abandonment = kpis.get("cart_abandonment_rate")
        return_rate = kpis.get("return_rate")
    
        # --- Conversion floor ---
        if cr is not None and cr < 0.015 and "Low Conversion Efficiency" not in used_titles:
            insights.append({
                "level": "WARNING",
                "title": "Low Conversion Efficiency",
                "so_what": f"Conversion rate is {cr:.2%}, indicating weak traffic-to-order efficiency.",
                "sub_domain": "conversion",
            })
    
        # --- Cart abandonment ---
        if abandonment is not None and abandonment > 0.70 and "High Cart Abandonment" not in used_titles:
            insights.append({
                "level": "WARNING",
                "title": "High Cart Abandonment",
                "so_what": f"{abandonment:.1%} of carts are abandoned before purchase.",
                "sub_domain": "conversion",
            })
    
        # --- Bounce ---
        if bounce is not None and bounce > 0.65 and "Traffic Quality Concern" not in used_titles:
            insights.append({
                "level": "WARNING",
                "title": "Traffic Quality Concern",
                "so_what": f"Bounce rate of {bounce:.1%} suggests low landing page relevance.",
                "sub_domain": "traffic",
            })
    
        # --- Returns ---
        if return_rate is not None and return_rate > 0.15 and "Elevated Return Rate" not in used_titles:
            insights.append({
                "level": "WARNING",
                "title": "Elevated Return Rate",
                "so_what": f"{return_rate:.1%} of orders are returned, indicating product or expectation mismatch.",
                "sub_domain": "operations",
            })
    
        if not insights:
            insights.append({
                "level": "INFO",
                "title": "E-Commerce Performance Stable",
                "so_what": "Traffic, conversion, and revenue signals are within healthy ranges.",
                "sub_domain": "mixed",
            })
    
        return insights
    
    
    # =====================================================
    # COMPOSITE INSIGHTS — SUB-DOMAIN RICH (≥7 EACH)
    # =====================================================
    
    def generate_composite_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []
    
        sessions = kpis.get("total_sessions", 0)
        users = kpis.get("total_users")
        cr = kpis.get("conversion_rate")
        abandonment = kpis.get("cart_abandonment_rate")
        checkout_drop = kpis.get("checkout_dropoff_rate")
        aov = kpis.get("aov")
        rps = kpis.get("revenue_per_session")
        repeat_rate = kpis.get("repeat_user_rate")
    
        # =================================================
        # TRAFFIC & ACQUISITION
        # =================================================
        if sessions > 1000 and cr is not None and cr < 0.01:
            insights.append({
                "level": "RISK",
                "title": "Inefficient Traffic Acquisition",
                "so_what": "High traffic volume is not translating into orders, indicating poor acquisition quality.",
                "sub_domain": "traffic",
            })
    
        if users and sessions and sessions / users > 3:
            insights.append({
                "level": "OPPORTUNITY",
                "title": "Repeat Visit Behavior Detected",
                "so_what": "Users are returning multiple times before converting, indicating remarketing opportunity.",
                "sub_domain": "traffic",
            })
    
        # =================================================
        # CONVERSION & FUNNEL
        # =================================================
        if abandonment and abandonment > 0.65:
            insights.append({
                "level": "WARNING",
                "title": "Cart Friction Dominates Funnel Loss",
                "so_what": "Most funnel drop-off occurs before checkout, signaling UX or pricing friction.",
                "sub_domain": "conversion",
            })
    
        if checkout_drop and checkout_drop > 0.70:
            insights.append({
                "level": "WARNING",
                "title": "Checkout Completion Barrier",
                "so_what": "A majority of users drop at checkout, suggesting payment or trust issues.",
                "sub_domain": "conversion",
            })
    
        if cr and cr > 0.03:
            insights.append({
                "level": "STRENGTH",
                "title": "Strong Conversion Efficiency",
                "so_what": "Conversion rate exceeds typical ecommerce benchmarks.",
                "sub_domain": "conversion",
            })
    
        # =================================================
        # REVENUE & ECONOMICS
        # =================================================
        if rps and rps > 2 * (aov or rps):
            insights.append({
                "level": "OPPORTUNITY",
                "title": "High Revenue Yield per Visit",
                "so_what": "Sessions are monetizing strongly, suggesting premium traffic quality.",
                "sub_domain": "revenue",
            })
    
        if aov and aov < 30:
            insights.append({
                "level": "OPPORTUNITY",
                "title": "AOV Expansion Potential",
                "so_what": "Low average order value suggests upsell and bundling opportunities.",
                "sub_domain": "revenue",
            })
    
        # =================================================
        # CUSTOMER & RETENTION
        # =================================================
        if repeat_rate and repeat_rate > 0.35:
            insights.append({
                "level": "STRENGTH",
                "title": "Healthy Repeat Purchase Behavior",
                "so_what": "A significant portion of users are returning customers.",
                "sub_domain": "customer",
            })
    
        if repeat_rate and repeat_rate < 0.15:
            insights.append({
                "level": "OPPORTUNITY",
                "title": "Low Customer Stickiness",
                "so_what": "Most customers purchase only once, indicating lifecycle improvement opportunity.",
                "sub_domain": "customer",
            })
    
        # =================================================
        # OPERATIONAL FRICTION
        # =================================================
        if kpis.get("return_rate") and kpis["return_rate"] > 0.20:
            insights.append({
                "level": "WARNING",
                "title": "Operational Leakage via Returns",
                "so_what": "High return rate is eroding realized revenue and margin.",
                "sub_domain": "operations",
            })
    
        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
    
        recs: List[Dict[str, Any]] = []
    
        cr = kpis.get("conversion_rate")
        abandonment = kpis.get("cart_abandonment_rate")
        checkout_drop = kpis.get("checkout_dropoff_rate")
        bounce = kpis.get("avg_bounce_rate")
        aov = kpis.get("aov")
        repeat_rate = kpis.get("repeat_user_rate")
        return_rate = kpis.get("return_rate")
    
        # =================================================
        # TRAFFIC & ACQUISITION
        # =================================================
        if bounce is not None and bounce > 0.65:
            recs.append({
                "action": "Refine landing page relevance and align ad messaging to reduce bounce.",
                "priority": "HIGH",
                "sub_domain": "traffic",
            })
    
        recs.append({
            "action": "Analyze source-level conversion to reallocate spend toward high-quality traffic.",
            "priority": "MEDIUM",
            "sub_domain": "traffic",
        })
    
        recs.append({
            "action": "Optimize mobile site performance to improve engagement from mobile users.",
            "priority": "LOW",
            "sub_domain": "traffic",
        })
    
        # =================================================
        # CONVERSION & FUNNEL
        # =================================================
        if cr is not None and cr < 0.02:
            recs.append({
                "action": "Simplify product pages and strengthen calls-to-action to improve conversion.",
                "priority": "HIGH",
                "sub_domain": "conversion",
            })
    
        if abandonment is not None and abandonment > 0.65:
            recs.append({
                "action": "Enable abandoned cart reminders and reduce checkout friction.",
                "priority": "HIGH",
                "sub_domain": "conversion",
            })
    
        if checkout_drop is not None and checkout_drop > 0.70:
            recs.append({
                "action": "Audit checkout flow for payment, trust, or UX barriers.",
                "priority": "HIGH",
                "sub_domain": "conversion",
            })
    
        recs.append({
            "action": "Introduce trust signals (reviews, guarantees) on checkout pages.",
            "priority": "MEDIUM",
            "sub_domain": "conversion",
        })
    
        # =================================================
        # REVENUE & ECONOMICS
        # =================================================
        if aov is not None and aov < 50:
            recs.append({
                "action": "Implement bundles and cross-sell recommendations to increase order value.",
                "priority": "MEDIUM",
                "sub_domain": "revenue",
            })
    
        recs.append({
            "action": "Test tiered pricing or volume discounts to improve revenue per order.",
            "priority": "LOW",
            "sub_domain": "revenue",
        })
    
        # =================================================
        # CUSTOMER & RETENTION
        # =================================================
        if repeat_rate is not None and repeat_rate < 0.20:
            recs.append({
                "action": "Launch post-purchase engagement campaigns to increase repeat buying.",
                "priority": "MEDIUM",
                "sub_domain": "customer",
            })
    
        recs.append({
            "action": "Introduce loyalty incentives for returning customers.",
            "priority": "LOW",
            "sub_domain": "customer",
            })
    
        # =================================================
        # OPERATIONAL FRICTION
        # =================================================
        if return_rate is not None and return_rate > 0.15:
            recs.append({
                "action": "Analyze return reasons to improve product descriptions and quality control.",
                "priority": "HIGH",
                "sub_domain": "operations",
            })
    
        return recs


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

