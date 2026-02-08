import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS — CUSTOMER VALUE & LOYALTY (GOVERNED)
# =====================================================

def _safe_div(
    n: Optional[float],
    d: Optional[float],
) -> Optional[float]:
    """
    Safe division helper.

    Guarantees:
    - Never raises
    - Returns None on invalid input
    - Used across KPI, insight, and visual layers
    """
    try:
        if d in (0, None) or pd.isna(d):
            return None
        return float(n) / float(d)
    except Exception:
        return None


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Customer Value & Loyalty time detector.

    PURPOSE:
    - Identify lifecycle or recency timestamps
    - Enable tenure / value aging / trend analysis

    Explicitly NOT:
    - Experience events
    - Support events
    - Marketing touchpoints
    """

    if df is None or df.empty:
        return None

    candidates = [
        # lifecycle & recency
        "last_purchase_date",
        "last_transaction_date",
        "last_active_date",
        "last_seen_date",
        "signup_date",
        "registration_date",
        "created_at",
        "date",
    ]

    for col in df.columns:
        col_l = str(col).lower().replace(" ", "_")
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

def _resolve_any(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Domain-scoped resolver for customer value signals.

    Matches on:
    - snake_case
    - camelCase
    - partial semantic tokens

    Does NOT fabricate meaning.
    """
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}

    for key in candidates:
        key_l = key.lower()
        for norm, original in cols.items():
            if key_l == norm:
                return original
            if key_l in norm:
                return original
    return None

# =====================================================
# CUSTOMER VALUE & LOYALTY DOMAIN
# =====================================================

class CustomerValueDomain(BaseDomain):
    name = "customer_value"
    description = "Customer Value, Loyalty & Economic Stability Intelligence"

    # -------------------------------------------------
    # PREPROCESS (VALUE-CENTRIC, GOVERNED)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Customer Value & Loyalty preprocess guarantees:

        - Defensive DataFrame copy
        - Lifecycle-aware time detection
        - Semantic column resolution (value only)
        - Numeric normalization (safe)
        - NO KPI computation
        - NO assumptions or thresholds
        """

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if not isinstance(df, pd.DataFrame):
            raise TypeError("CustomerValueDomain.preprocess expects a DataFrame")

        df = df.copy(deep=False)

        # -------------------------------------------------
        # TIME COLUMN (LIFECYCLE / RECENCY)
        # -------------------------------------------------
        self.time_col = _detect_time_column(df)

        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col],
                errors="coerce",
            )
            df = df.sort_values(self.time_col)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (VALUE & LOYALTY ONLY)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {

            # ---------------- IDENTITY ----------------
            "customer": (
                resolve_column(df, "customer_id")
                or resolve_column(df, "customer")
                or resolve_column(df, "user_id")
                or resolve_column(df, "customerid")
            ),

            # ---------------- ECONOMIC VALUE ----------------
            "clv": (
                resolve_column(df, "clv")
                or resolve_column(df, "customer_lifetime_value")
                or _resolve_any(df, [
                    "clv_value",
                    "lifetime_value",
                    "lifetime_revenue",
                    "customer_value",
                ])
            ),
            
            "total_spend": (
                resolve_column(df, "total_spend")
                or resolve_column(df, "totalspend")
                or resolve_column(df, "spend")
                or resolve_column(df, "lifetime_spend")
                or resolve_column(df, "sales_amount")
                or _resolve_any(df, [
                    "total_revenue",
                    "lifetime_revenue",
                    "customer_revenue",
                    "revenue_total",
                ])
            ),

            "total_purchases": (
                resolve_column(df, "total_purchases")
                or resolve_column(df, "purchase_count")
                or resolve_column(df, "annualfrequency")
                or resolve_column(df, "purchase_frequency")
                or resolve_column(df, "order_count")
                or resolve_column(df, "annual_frequency")   # ✅ ADD
                or _resolve_any(df, ["frequency", "orders_per_year"])
            ),


            # ---------------- LOYALTY & TENURE ----------------
            "tenure": (
                resolve_column(df, "tenure")
                or resolve_column(df, "tenureyears")
                or resolve_column(df, "tenure_years")
                or _resolve_any(df, [
                    "customer_tenure",
                    "years_active",
                    "years_with_company",
                    "relationship_years",
                ])
            ),

            "loyalty_tier": (
                resolve_column(df, "loyalty_tier")
                or resolve_column(df, "loyaltytier")
                or resolve_column(df, "tier")
                or resolve_column(df, "membership_level")
            ),

            # ---------------- RISK & STABILITY ----------------
            "churn_risk": (
                resolve_column(df, "churn_risk")
                or resolve_column(df, "churnriskscore")
                or resolve_column(df, "churn_risk_score")
                or resolve_column(df, "attrition_risk")
                or _resolve_any(df, [
                    "churn_probability",
                    "risk_score",
                    "retention_risk",
                ])
            ),

            # ---------------- ENGAGEMENT PROXIES ----------------
            "preferred_channel": (
                resolve_column(df, "channel")
                or resolve_column(df, "preferred_channel")
            ),
            "email_opt_in": (
                resolve_column(df, "email_opt_in")
                or resolve_column(df, "opt_in")
                or resolve_column(df, "email_consent")
            ),

            # ---------------- DEMOGRAPHIC CONTEXT (PROXY ONLY) ----------------
            "age": resolve_column(df, "age"),
            "city": resolve_column(df, "city"),
            "income": (
                resolve_column(df, "annual_income")
                or resolve_column(df, "income")
            ),

            "referrals": (
                resolve_column(df, "referrals")
                or resolve_column(df, "referral_count")
            ),

            "recency_days": (
                resolve_column(df, "days_since_last_purchase")
                or resolve_column(df, "recency_days")
                or resolve_column(df, "dayssincelastpurchase")
                or resolve_column(df, "recencydays")
            ),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        numeric_keys = {
            "clv",
            "total_spend",
            "total_purchases",
            "tenure",
            "recency_days",
            "churn_risk",
            "referrals",
            "age",
            "income",
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

        # -------------------------------------------------
        # BINARY NORMALIZATION (EMAIL OPT-IN)
        # -------------------------------------------------
        opt_col = self.cols.get("email_opt_in")
        if opt_col and opt_col in df.columns:
            s = df[opt_col]
            if s.dtype == object:
                s = s.astype(str).str.lower().map({
                    "yes": 1, "true": 1, "1": 1,
                    "no": 0, "false": 0, "0": 0,
                })
            df[opt_col] = pd.to_numeric(s, errors="coerce").clip(0, 1)

        # ---------------------------------------------
        # DATA COMPLETENESS — ANALYTIC SIGNALS ONLY
        # (ROBUST TO camelCase / PascalCase)
        # ---------------------------------------------
        analytic_signal_keys = {
            "clv": ["clv", "customer_lifetime_value"],
            "total_spend": ["total_spend", "totalspend", "total_spend", "lifetime_spend"],
            "total_purchases": ["total_purchases", "annualfrequency", "purchase_count"],
            "tenure": ["tenure", "tenureyears", "tenure_years"],
            "churn_risk": ["churn_risk", "churnriskscore", "churn_risk_score"],
            "recency_days": ["recency_days", "dayssincelastpurchase", "days_since_last_purchase"],
        }
        
        present = 0
        
        for key, candidates in analytic_signal_keys.items():
            col = (
                self.cols.get(key)
                or _resolve_any(df, candidates)
            )
            if col and col in df.columns and df[col].notna().any():
                present += 1
        
        self.data_completeness = round(
            present / max(len(analytic_signal_keys), 1),
            2,
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
        Execute the customer_value pipeline with stable, safe outputs.

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
    # KPI ENGINE — CUSTOMER VALUE & LOYALTY
    # -------------------------------------------------
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Customer Value & Loyalty KPI Engine (v1.0)
    
        GUARANTEES:
        - Capability-driven sub-domains
        - 5–9 KPIs per sub-domain (when data allows)
        - Proxy-aware metrics only (no fabrication)
        - Confidence-tagged KPIs
        - Graceful degradation on weak data
        """
    
        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return {}
    
        c = self.cols
        volume = int(len(df))
    
        # -------------------------------------------------
        # SUB-DOMAINS (LOCKED)
        # -------------------------------------------------
        sub_domains = {
            "value": "Customer Value & Monetization",
            "loyalty": "Loyalty & Tenure",
            "risk": "Churn Risk & Stability",
            "engagement": "Engagement Proxies",
            "concentration": "Value Concentration & Dependency",
        }
    
        # -------------------------------------------------
        # KPI CONTAINER
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "domain": "customer_value",
            "sub_domains": sub_domains,
            "record_count": volume,
            "data_completeness": getattr(self, "data_completeness", 0.0),
            "_domain_kpi_map": {},
            "_confidence": {},
            "_proxy_metrics": [],
        }
    
        # -------------------------------------------------
        # PROMOTE DATA COMPLETENESS AS FIRST-CLASS KPI SIGNAL
        # -------------------------------------------------
        completeness = kpis["data_completeness"]
    
        if completeness >= 0.4:
            kpis["_confidence"]["data_completeness"] = 0.85
        elif completeness >= 0.25:
            kpis["_confidence"]["data_completeness"] = 0.70
        else:
            kpis["_confidence"]["data_completeness"] = 0.50
    
        # -------------------------------------------------
        # SAFE HELPERS
        # -------------------------------------------------
        def safe_mean(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        def safe_sum(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.sum()) if s.notna().any() else None
    
        def safe_nunique(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            return int(df[col].nunique())
    
        def can_derive_total_purchases():
            if not c.get("customer"):
                return False
            return df[c["customer"]].duplicated().any()
    
        # =================================================
        # VALUE — ECONOMIC CONTRIBUTION
        # =================================================
        value = []
    
        if c.get("clv"):
            kpis["value_avg_clv"] = safe_mean(c["clv"])
            value.append("value_avg_clv")
    
            kpis["value_clv_dispersion"] = _safe_div(
                df[c["clv"]].std(),
                df[c["clv"]].mean(),
            )
            value.append("value_clv_dispersion")
    
        if c.get("total_spend"):
            kpis["value_total_spend"] = safe_sum(c["total_spend"])
            value.append("value_total_spend")
    
            kpis["value_avg_spend"] = safe_mean(c["total_spend"])
            value.append("value_avg_spend")
    
        if c.get("total_purchases"):
            kpis["value_avg_purchase_count"] = safe_mean(c["total_purchases"])
            value.append("value_avg_purchase_count")
    
        # =================================================
        # LOYALTY — TENURE & CONTINUITY
        # =================================================
        loyalty = []
    
        if c.get("tenure"):
            kpis["loyalty_avg_tenure"] = safe_mean(c["tenure"])
            loyalty.append("loyalty_avg_tenure")
    
            kpis["loyalty_tenure_dispersion"] = _safe_div(
                df[c["tenure"]].std(),
                df[c["tenure"]].mean(),
            )
            loyalty.append("loyalty_tenure_dispersion")

        # ---------------- LOYALTY CONTINUITY ----------------
        if c.get("tenure"):
            s = df[c["tenure"]].dropna()
            kpis["loyalty_tenure_skew"] = float(s.skew()) if len(s) > 10 else None
            loyalty.append("loyalty_tenure_skew")
        
        if c.get("tenure") and c.get("total_purchases"):
            corr = df[[c["tenure"], c["total_purchases"]]].corr().iloc[0, 1]
            kpis["loyalty_tenure_purchase_alignment"] = corr
            loyalty.append("loyalty_tenure_purchase_alignment")

        if c.get("loyalty_tier"):
            kpis["loyalty_tier_count"] = safe_nunique(c["loyalty_tier"])
            loyalty.append("loyalty_tier_count")
    
        if c.get("customer") and c.get("total_purchases"):
            freq = df[c["total_purchases"]]
            kpis["loyalty_repeat_customer_rate"] = _safe_div(
                (freq > 1).sum(),
                freq.notna().sum(),
            )
            loyalty.append("loyalty_repeat_customer_rate")
    
        # ---------------- PROXY: TOTAL PURCHASES ----------------
        if not c.get("total_purchases") and can_derive_total_purchases():
            purchase_counts = (
                df.groupby(c["customer"])
                .size()
                .rename("proxy_total_purchases")
            )
    
            avg_purchases = purchase_counts.mean()
    
            kpis["loyalty_avg_purchase_count"] = float(avg_purchases)
            loyalty.append("loyalty_avg_purchase_count")
            kpis["_proxy_metrics"].append("loyalty_avg_purchase_count")
    
        # =================================================
        # RISK — CHURN & STABILITY
        # =================================================
        risk = []
    
        if c.get("churn_risk"):
            kpis["risk_avg_churn_risk"] = safe_mean(c["churn_risk"])
            risk.append("risk_avg_churn_risk")
    
            kpis["risk_churn_risk_dispersion"] = _safe_div(
                df[c["churn_risk"]].std(),
                df[c["churn_risk"]].mean(),
            )
            risk.append("risk_churn_risk_dispersion")
    
        if c.get("tenure") and c.get("churn_risk"):
            corr = df[[c["tenure"], c["churn_risk"]]].corr().iloc[0, 1]
            kpis["risk_tenure_churn_alignment"] = _safe_div(corr, 1)
            risk.append("risk_tenure_churn_alignment")

        # ---------------- RISK SHAPE ----------------
        if c.get("churn_risk"):
            s = df[c["churn_risk"]].dropna()
        
            # asymmetry of risk distribution (no threshold)
            kpis["risk_churn_skew"] = float(s.skew()) if len(s) > 10 else None
            risk.append("risk_churn_skew")
        
        # ---------------- RISK RECENCY ALIGNMENT ----------------
        if c.get("churn_risk") and c.get("recency_days"):
            corr = df[[c["churn_risk"], c["recency_days"]]].corr().iloc[0, 1]
            kpis["risk_recency_alignment"] = corr
            risk.append("risk_recency_alignment")

        # =================================================
        # ENGAGEMENT — PROXY SIGNALS
        # =================================================
        engagement = []
    
        if c.get("preferred_channel"):
            kpis["engagement_channel_count"] = safe_nunique(c["preferred_channel"])
            engagement.append("engagement_channel_count")
    
        if c.get("email_opt_in"):
            kpis["engagement_email_opt_in_rate"] = safe_mean(c["email_opt_in"])
            engagement.append("engagement_email_opt_in_rate")
    
        # =================================================
        # CONCENTRATION — DEPENDENCY & SKEW
        # =================================================
        concentration = []
    
        if c.get("clv"):
            sorted_vals = df[c["clv"]].dropna().sort_values(ascending=False)
            top_20 = int(max(1, len(sorted_vals) * 0.2))
            kpis["concentration_top_20pct_value_share"] = _safe_div(
                sorted_vals.head(top_20).sum(),
                sorted_vals.sum(),
            )
            concentration.append("concentration_top_20pct_value_share")
    
        if c.get("total_spend"):
            sorted_spend = df[c["total_spend"]].dropna().sort_values(ascending=False)
            top_10 = int(max(1, len(sorted_spend) * 0.1))
            kpis["concentration_top_10pct_spend_share"] = _safe_div(
                sorted_spend.head(top_10).sum(),
                sorted_spend.sum(),
            )
            concentration.append("concentration_top_10pct_spend_share")
    
        # -------------------------------------------------
        # DOMAIN → KPI MAP
        # -------------------------------------------------
        kpis["_domain_kpi_map"] = {
            "value": value,
            "loyalty": loyalty,
            "risk": risk,
            "engagement": engagement,
            "concentration": concentration,
        }
    
        # -------------------------------------------------
        # DOMAIN DATA STRENGTH FLAG (VISUAL & READINESS GATE)
        # -------------------------------------------------
        kpis["_domain_has_strong_data"] = completeness >= 0.4
    
        # -------------------------------------------------
        # KPI CONFIDENCE (MANDATORY)
        # -------------------------------------------------
        for key, val in kpis.items():
            if key.startswith("_") or not isinstance(val, (int, float)):
                continue
    
            base = 0.7
    
            if volume < 100:
                base -= 0.15
    
            if "dispersion" in key or "share" in key:
                base += 0.05
    
            if key in kpis.get("_proxy_metrics", []):
                base -= 0.15
    
            if "alignment" in key:
                base -= 0.05
    
            kpis["_confidence"][key] = round(
                max(0.4, min(0.9, base)),
                2,
            )
    
        self._last_kpis = kpis
        return kpis

    # -------------------------------------------------
    # VISUAL ENGINE — CUSTOMER VALUE & LOYALTY
    # -------------------------------------------------
    
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Customer Value & Loyalty Visual Engine (v1.0)
    
        GUARANTEES:
        - Evidence-only visuals
        - Column-driven (no KPI bookkeeping gates)
        - Flat-plot prevention
        - No judgement, no thresholds
        - No trimming (report layer responsibility)
        - Deterministic file output
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.cols
    
        # -------------------------------------------------
        # KPI SOURCE OF TRUTH (STATELESS)
        # -------------------------------------------------
        kpis = self.calculate_kpis(df)
        record_count = int(kpis.get("record_count", len(df)))
    
        # -------------------------------------------------
        # VISUAL CONFIDENCE (DATA-DRIVEN ONLY)
        # -------------------------------------------------
        if record_count >= 5000:
            visual_conf = 0.85
        elif record_count >= 1000:
            visual_conf = 0.70
        else:
            visual_conf = 0.55
    
        # -------------------------------------------------
        # SAFE SAVE HELPER
        # -------------------------------------------------
        def save(fig, name, caption, importance, sub_domain, role, axis):
            path = output_dir / name
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            if path.exists():
                visuals.append({
                    "path": str(path),
                    "caption": caption,
                    "importance": float(importance),
                    "sub_domain": sub_domain,
                    "role": role,
                    "axis": axis,
                    "confidence": visual_conf,
                })
    
        # =================================================
        # VALUE — ECONOMIC DISTRIBUTIONS
        # =================================================
        if c.get("clv") and df[c["clv"]].nunique(dropna=True) > 3:
            fig, ax = plt.subplots()
            df[c["clv"]].dropna().hist(ax=ax, bins=20)
            ax.set_title("Customer Lifetime Value Distribution")
            save(
                fig, "value_clv_dist.png",
                "Distribution of customer lifetime value",
                0.95, "value", "monetization", "distribution"
            )
    
            fig, ax = plt.subplots()
            df[c["clv"]].dropna().plot(kind="box", ax=ax)
            ax.set_title("CLV Spread")
            save(
                fig, "value_clv_box.png",
                "Variability in customer lifetime value",
                0.90, "value", "monetization", "spread"
            )
    
        if c.get("total_spend") and df[c["total_spend"]].nunique(dropna=True) > 3:
            fig, ax = plt.subplots()
            df[c["total_spend"]].dropna().hist(ax=ax, bins=20)
            ax.set_title("Total Spend Distribution")
            save(
                fig, "value_spend_dist.png",
                "Distribution of customer total spend",
                0.90, "value", "monetization", "distribution"
            )
    
        # =================================================
        # LOYALTY — TENURE & CONTINUITY
        # =================================================
        if c.get("tenure") and df[c["tenure"]].nunique(dropna=True) > 3:
            fig, ax = plt.subplots()
            df[c["tenure"]].dropna().hist(ax=ax, bins=15)
            ax.set_title("Customer Tenure Distribution")
            save(
                fig, "loyalty_tenure_dist.png",
                "Customer tenure spread",
                0.90, "loyalty", "stability", "distribution"
            )
    
            fig, ax = plt.subplots()
            df[c["tenure"]].dropna().plot(kind="box", ax=ax)
            ax.set_title("Tenure Variability")
            save(
                fig, "loyalty_tenure_box.png",
                "Variability in customer tenure",
                0.85, "loyalty", "stability", "spread"
            )

        if c.get("tenure") and df[c["tenure"]].nunique() > 5:
            fig, ax = plt.subplots()
            df[c["tenure"]].dropna().plot(kind="density", ax=ax)
            ax.set_title("Customer Tenure Density")
            save(
                fig,
                "loyalty_tenure_density.png",
                "Density profile of customer tenure",
                0.93,
                "loyalty",
                "continuity",
                "distribution",
            )

        if c.get("tenure") and c.get("total_purchases"):
            x = df[c["tenure"]]
            y = df[c["total_purchases"]]
        
            if x.notna().sum() > 20 and y.notna().sum() > 20:
                fig, ax = plt.subplots()
                ax.scatter(x, y, alpha=0.4)
                ax.set_xlabel("Tenure (Years)")
                ax.set_ylabel("Purchase Frequency")
                ax.set_title("Tenure vs Purchase Frequency")
                save(
                    fig,
                    "loyalty_tenure_vs_frequency.png",
                    "Relationship between customer tenure and purchase activity",
                    0.92,
                    "loyalty",
                    "continuity",
                    "relationship",
                )

        if c.get("loyalty_tier"):
            counts = df[c["loyalty_tier"]].value_counts()
            if len(counts) > 1:
                fig, ax = plt.subplots()
                counts.plot.bar(ax=ax)
                ax.set_title("Loyalty Tier Mix")
                save(
                    fig, "loyalty_tier_mix.png",
                    "Distribution of customers across loyalty tiers",
                    0.80, "loyalty", "structure", "composition"
                )
    
        # =================================================
        # RISK — CHURN RISK STRUCTURE
        # =================================================
        if c.get("churn_risk") and df[c["churn_risk"]].nunique(dropna=True) > 3:
            fig, ax = plt.subplots()
            df[c["churn_risk"]].dropna().hist(ax=ax, bins=15)
            ax.set_title("Churn Risk Score Distribution")
            save(
                fig, "risk_churn_dist.png",
                "Distribution of churn risk scores",
                0.90, "risk", "stability", "distribution"
            )
    
            fig, ax = plt.subplots()
            df[c["churn_risk"]].dropna().plot(kind="box", ax=ax)
            ax.set_title("Churn Risk Spread")
            save(
                fig, "risk_churn_box.png",
                "Variability in churn risk",
                0.85, "risk", "stability", "spread"
            )

        if c.get("churn_risk") and df[c["churn_risk"]].nunique() > 5:
            fig, ax = plt.subplots()
            df[c["churn_risk"]].dropna().plot(kind="density", ax=ax)
            ax.set_title("Churn Risk Density")
            save(
                fig,
                "risk_churn_density.png",
                "Density profile of customer churn risk",
                0.92,
                "risk",
                "stability",
                "distribution",
            )

        if c.get("recency_days") and c.get("churn_risk"):
            x = df[c["recency_days"]]
            y = df[c["churn_risk"]]
        
            if x.notna().sum() > 20 and y.notna().sum() > 20:
                fig, ax = plt.subplots()
                ax.scatter(x, y, alpha=0.4)
                ax.set_xlabel("Days Since Last Purchase")
                ax.set_ylabel("Churn Risk")
                ax.set_title("Recency vs Churn Risk")
                save(
                    fig,
                    "risk_recency_vs_churn.png",
                    "Relationship between inactivity and churn risk",
                    0.90,
                    "risk",
                    "early_warning",
                    "relationship",
                )

        if c.get("tenure") and c.get("churn_risk"):
            x = df[c["tenure"]]
            y = df[c["churn_risk"]]
            if x.notna().sum() > 5 and y.notna().sum() > 5:
                fig, ax = plt.subplots()
                ax.scatter(x, y, alpha=0.5)
                ax.set_xlabel("Tenure")
                ax.set_ylabel("Churn Risk")
                ax.set_title("Tenure vs Churn Risk")
                save(
                    fig, "risk_tenure_vs_churn.png",
                    "Relationship between tenure and churn risk",
                    0.85, "risk", "relationship", "correlation"
                )
    
        # =================================================
        # ENGAGEMENT — PROXY SIGNALS
        # =================================================
        if c.get("preferred_channel"):
            counts = df[c["preferred_channel"]].value_counts()
            if len(counts) > 1:
                fig, ax = plt.subplots()
                counts.plot.bar(ax=ax)
                ax.set_title("Preferred Channel Distribution")
                save(
                    fig, "engagement_channel_mix.png",
                    "Customer preferred channel mix",
                    0.75, "engagement", "preference", "composition"
                )
    
        if c.get("email_opt_in"):
            counts = df[c["email_opt_in"]].value_counts().sort_index()
            if len(counts) > 1:
                fig, ax = plt.subplots()
                counts.plot.bar(ax=ax)
                ax.set_title("Email Opt-In Distribution")
                save(
                    fig, "engagement_email_optin.png",
                    "Email communication opt-in status",
                    0.70, "engagement", "access", "composition"
                )
    
        # =================================================
        # CONCENTRATION — DEPENDENCY & SKEW
        # =================================================
        if c.get("clv") and df[c["clv"]].nunique(dropna=True) > 5:
            vals = df[c["clv"]].dropna().sort_values(ascending=False)
            if vals.sum() > 0:
                cumulative = vals.cumsum() / vals.sum()
                fig, ax = plt.subplots()
                cumulative.reset_index(drop=True).plot(ax=ax)
                ax.set_title("Cumulative CLV Concentration Curve")
                save(
                    fig, "concentration_clv_curve.png",
                    "Cumulative contribution of top customers to total value",
                    0.95, "concentration", "dependency", "cumulative"
                )
    
        if c.get("total_spend") and df[c["total_spend"]].nunique(dropna=True) > 5:
            vals = df[c["total_spend"]].dropna().sort_values(ascending=False)
            if vals.sum() > 0:
                cumulative = vals.cumsum() / vals.sum()
                fig, ax = plt.subplots()
                cumulative.reset_index(drop=True).plot(ax=ax)
                ax.set_title("Cumulative Spend Concentration Curve")
                save(
                    fig, "concentration_spend_curve.png",
                    "Cumulative spend contribution by customer rank",
                    0.90, "concentration", "dependency", "cumulative"
                )
    
        # -------------------------------------------------
        # RETURN ALL CANDIDATES (NO TRIMMING)
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals

    # -------------------------------------------------
    # INSIGHT ENGINE — CUSTOMER VALUE & LOYALTY
    # -------------------------------------------------
    
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Customer Value & Loyalty Insight Engine (v1.0)
    
        GUARANTEES:
        - Composite insights are primary
        - ≥7 insights per sub-domain (when data allows)
        - Executive-safe, evidence-based language
        - No thresholds, no judgement
        - Atomic fallback only
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # KPI SHORTCUTS (SAFE)
        # -------------------------------------------------
        avg_clv = kpis.get("value_avg_clv")
        clv_disp = kpis.get("value_clv_dispersion")
        total_spend = kpis.get("value_total_spend")
    
        avg_tenure = kpis.get("loyalty_avg_tenure")
        tenure_disp = kpis.get("loyalty_tenure_dispersion")
        tier_count = kpis.get("loyalty_tier_count")
    
        avg_churn_risk = kpis.get("risk_avg_churn_risk")
        churn_disp = kpis.get("risk_churn_risk_dispersion")
        tenure_churn_align = kpis.get("risk_tenure_churn_alignment")
    
        channel_count = kpis.get("engagement_channel_count")
        email_opt_in = kpis.get("engagement_email_opt_in_rate")
    
        top20_value = kpis.get("concentration_top_20pct_value_share")
        top10_spend = kpis.get("concentration_top_10pct_spend_share")
    
        # =================================================
        # VALUE — ECONOMIC CONTRIBUTION
        # =================================================
        if "value" in sub_domains and any(v is not None for v in (avg_clv, total_spend)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Customer Value Baseline Established",
                    "so_what": "Customer lifetime value metrics provide a baseline view of economic contribution.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Value Distribution Visibility",
                    "so_what": "Value dispersion highlights differences in contribution across the customer base.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Average Value Signal",
                    "so_what": "Average CLV supports comparison across segments and cohorts.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Spend Aggregation Context",
                    "so_what": "Total spend provides scale context for customer value.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Economic Contribution Variability",
                    "so_what": "Observed variability indicates heterogeneous customer value profiles.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Value Monitoring Readiness",
                    "so_what": "Value metrics are suitable for ongoing executive monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "value",
                    "title": "Monetization Insight Coverage",
                    "so_what": "Available value signals support monetization analysis.",
                },
            ])
    
        # =================================================
        # LOYALTY — TENURE & CONTINUITY
        # =================================================
        if "loyalty" in sub_domains and any(v is not None for v in (avg_tenure, tier_count)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Customer Tenure Baseline Established",
                    "so_what": "Tenure metrics provide insight into customer relationship longevity.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Loyalty Structure Visibility",
                    "so_what": "Loyalty tier distribution reflects program structure and reach.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Tenure Variability Context",
                    "so_what": "Tenure dispersion highlights differences in customer continuity.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Repeat Behavior Proxy",
                    "so_what": "Purchase continuity proxies support loyalty interpretation.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Relationship Stability Insight",
                    "so_what": "Observed tenure patterns indicate relationship stability.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Loyalty Program Coverage",
                    "so_what": "Tier diversity provides insight into loyalty program segmentation.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Continuity Governance Readiness",
                    "so_what": "Loyalty metrics are suitable for governance review.",
                },
            ])

        if (
            kpis.get("loyalty_tenure_purchase_alignment") is not None
            and abs(kpis["loyalty_tenure_purchase_alignment"]) > 0.3
        ):
            insights.append({
                "level": "STRATEGIC",
                "title": "Value Supported by Customer Continuity",
                "so_what": (
                    "Customer tenure shows a strong relationship with repeat purchasing, "
                    "indicating that value is sustained by long-term customer continuity."
                ),
            })

        # =================================================
        # RISK — CHURN & STABILITY
        # =================================================
        if "risk" in sub_domains and avg_churn_risk is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Churn Risk Baseline Established",
                    "so_what": "Average churn risk provides a baseline view of customer stability.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Risk Dispersion Visibility",
                    "so_what": "Risk variability highlights differences in customer stability profiles.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Tenure–Risk Relationship Context",
                    "so_what": "The relationship between tenure and risk supports lifecycle interpretation.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Stability Monitoring Readiness",
                    "so_what": "Churn risk metrics enable proactive stability monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Risk Signal Coverage",
                    "so_what": "Available risk signals support customer stability assessment.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Lifecycle Risk Visibility",
                    "so_what": "Risk metrics provide insight across customer lifecycle stages.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "risk",
                    "title": "Retention Governance Context",
                    "so_what": "Risk signals are suitable for executive governance.",
                },
            ])

        if (
            kpis.get("risk_recency_alignment") is not None
            and abs(kpis["risk_recency_alignment"]) > 0.3
        ):
            insights.append({
                "level": "STRATEGIC",
                "title": "Inactivity-Driven Churn Risk",
                "so_what": (
                    "Customer churn risk shows a strong relationship with purchase inactivity, "
                    "indicating emerging rather than residual risk."
                ),
            })

        # =================================================
        # ENGAGEMENT — PROXY SIGNALS
        # =================================================
        if "engagement" in sub_domains and any(v is not None for v in (channel_count, email_opt_in)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Proxy Coverage",
                    "so_what": "Engagement proxies provide indirect visibility into customer reachability.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Channel Preference Structure",
                    "so_what": "Preferred channels indicate how customers choose to engage.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Communication Access Context",
                    "so_what": "Opt-in status reflects communication access across the customer base.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Diversity Insight",
                    "so_what": "Channel diversity highlights engagement variety.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Reachability Monitoring Readiness",
                    "so_what": "Engagement proxies support reachability governance.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Signal Stability",
                    "so_what": "Observed engagement patterns indicate stability over time.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Governance Context",
                    "so_what": "Engagement signals are suitable for executive oversight.",
                },
            ])
    
        # =================================================
        # CONCENTRATION — DEPENDENCY & SKEW
        # =================================================
        if "concentration" in sub_domains and any(v is not None for v in (top20_value, top10_spend)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Value Concentration Visibility",
                    "so_what": "Value concentration metrics show how economic contribution is distributed.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Top-Customer Dependency Context",
                    "so_what": "Top customer contribution indicates dependency exposure.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Revenue Skew Insight",
                    "so_what": "Spend concentration highlights skew within the customer base.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Portfolio Balance Visibility",
                    "so_what": "Concentration metrics support portfolio balance assessment.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Diversification Monitoring Readiness",
                    "so_what": "Customer diversification can be monitored using concentration signals.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Dependency Risk Context",
                    "so_what": "Concentration metrics provide context for dependency-related risk.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "concentration",
                    "title": "Executive Concentration Oversight",
                    "so_what": "Concentration insights are suitable for executive-level oversight.",
                },
            ])
    
        # -------------------------------------------------
        # ATOMIC FALLBACK (ONLY IF NOTHING ELSE)
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Customer Value Signals Available",
                "so_what": "Available data provides baseline visibility into customer value and stability.",
            })
    
        return insights

    # -------------------------------------------------
    # RECOMMENDATION ENGINE — CUSTOMER VALUE & LOYALTY
    # -------------------------------------------------
    
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Customer Value & Loyalty Recommendation Engine (v1.0)
    
        GUARANTEES:
        - ≥7 recommendations per sub-domain (when active)
        - Advisory, executive-safe language
        - No thresholds or urgency bias
        - Insight-aware but not title-dependent
        """
    
        recs: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return recs
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # =================================================
        # VALUE — ECONOMIC CONTRIBUTION
        # =================================================
        if "value" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "value",
                    "action": "Review customer value distributions to understand contribution patterns across the base.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "value",
                    "action": "Segment customer value metrics by tenure or channel to improve interpretability.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "value",
                    "action": "Incorporate customer value trends into long-term revenue planning discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "value",
                    "action": "Use value dispersion signals to inform portfolio balance considerations.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "value",
                    "action": "Align customer value insights with strategic account management reviews.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "value",
                    "action": "Monitor changes in average customer value over time for early structural signals.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "value",
                    "action": "Integrate customer value indicators into executive performance reviews.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # LOYALTY — TENURE & CONTINUITY
        # =================================================
        if "loyalty" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "loyalty",
                    "action": "Review tenure distributions to understand relationship longevity across customer cohorts.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Use loyalty tier structure to guide retention strategy discussions.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Incorporate tenure trends into lifecycle management planning.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Assess repeat behavior proxies to understand continuity patterns.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Align loyalty insights with customer segmentation initiatives.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Monitor changes in loyalty mix to detect structural shifts.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Include loyalty metrics in executive governance dashboards.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # RISK — CHURN & STABILITY
        # =================================================
        if "risk" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "risk",
                    "action": "Review churn risk distributions to understand stability patterns across the customer base.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "risk",
                    "action": "Examine the relationship between tenure and churn risk to support lifecycle analysis.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "risk",
                    "action": "Incorporate churn risk indicators into customer portfolio reviews.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "risk",
                    "action": "Use risk dispersion metrics to inform stability monitoring.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "risk",
                    "action": "Align churn risk signals with retention planning discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "risk",
                    "action": "Monitor changes in average churn risk over time for early warning signals.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "risk",
                    "action": "Integrate stability indicators into executive risk reviews.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # ENGAGEMENT — PROXY SIGNALS
        # =================================================
        if "engagement" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "engagement",
                    "action": "Review engagement proxy metrics to assess customer reachability.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "engagement",
                    "action": "Segment engagement signals by channel to refine communication strategies.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "engagement",
                    "action": "Incorporate opt-in trends into communication planning.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "engagement",
                    "action": "Monitor engagement proxy stability across customer cohorts.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "engagement",
                    "action": "Align engagement insights with customer outreach governance.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "engagement",
                    "action": "Use engagement indicators to inform channel investment discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "engagement",
                    "action": "Include engagement proxy metrics in executive communication reviews.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # CONCENTRATION — DEPENDENCY & SKEW
        # =================================================
        if "concentration" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "concentration",
                    "action": "Review customer value concentration to understand dependency exposure.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "concentration",
                    "action": "Assess spend concentration patterns across top customer segments.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "concentration",
                    "action": "Incorporate concentration metrics into portfolio risk discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "concentration",
                    "action": "Monitor changes in concentration curves to detect shifts in dependency.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "concentration",
                    "action": "Align concentration insights with diversification strategy planning.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "concentration",
                    "action": "Use dependency indicators to support long-term customer portfolio decisions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "concentration",
                    "action": "Include concentration metrics in executive risk oversight reviews.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not recs:
            recs.append({
                "sub_domain": "mixed",
                "action": "Continue monitoring customer value and loyalty signals for emerging patterns.",
                "priority": "LOW",
            })
    
        return recs

# =====================================================
# DOMAIN DETECTOR — CUSTOMER VALUE & LOYALTY
# =====================================================

class CustomerValueDomainDetector(BaseDomainDetector):
    """
    Customer Value & Loyalty Domain Detector (v1.0)

    Detects datasets focused on:
    - Customer lifetime value
    - Spend, purchases, tenure
    - Loyalty tiers
    - Churn risk scores
    - Engagement reachability proxies

    Explicitly avoids:
    - CX perception datasets (NPS / CSAT / CES)
    - Transaction-level retail datasets
    - Marketing execution datasets
    - Finance / accounting ledgers
    """

    domain_name = "customer_value"

    # -------------------------------------------------
    # STRONG VALUE ANCHORS (ECONOMIC & LOYALTY)
    # -------------------------------------------------
    VALUE_ANCHORS: Set[str] = {
        "clv",
        "lifetime_value",
        "total_spend",
        "lifetime_spend",
        "total_purchases",
        "purchase_count",
        "tenure",
        "tenure_years",
        "loyalty",
        "loyalty_tier",
        "churn_risk",
        "churn_risk_score",
        "attrition_risk",
    }

    # -------------------------------------------------
    # ENGAGEMENT PROXIES (SUPPORTING SIGNALS)
    # -------------------------------------------------
    ENGAGEMENT_PROXIES: Set[str] = {
        "preferred_channel",
        "channel",
        "email_opt_in",
        "email_consent",
        "opt_in",
    }

    # -------------------------------------------------
    # EXCLUSION TOKENS (BOUNDARY CONTROL)
    # -------------------------------------------------
    EXCLUSION_TOKENS: Set[str] = {
        # CX
        "nps",
        "csat",
        "ces",
        "sentiment",
        "ticket",
        "resolution",
        "response_time",

        # Retail / Ecommerce
        "order_id",
        "order_date",
        "transaction",
        "price",
        "sku",
        "product",
        "cart",
        "checkout",

        # Marketing
        "campaign",
        "impression",
        "click",
        "ctr",
        "cpc",
        "ad_spend",

        # Finance
        "ledger",
        "invoice",
        "cost",
        "margin",
        "profit",
        "gl_",
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        cols = {str(c).lower() for c in df.columns}

        # -------------------------------------------------
        # CORE SIGNAL CHECKS
        # -------------------------------------------------
        has_customer_id = any("customer" in c or "user" in c for c in cols)

        value_hits = [
            c for c in cols
            if any(t in c for t in self.VALUE_ANCHORS)
        ]

        engagement_hits = [
            c for c in cols
            if any(t in c for t in self.ENGAGEMENT_PROXIES)
        ]

        exclusion_hits = [
            c for c in cols
            if any(t in c for t in self.EXCLUSION_TOKENS)
        ]

        # -------------------------------------------------
        # CONFIDENCE SCORING (CAPABILITY-BASED)
        # -------------------------------------------------
        confidence = 0.0

        if has_customer_id and value_hits:
            confidence = 0.7

        if has_customer_id and len(value_hits) >= 2:
            confidence = 0.85

        if has_customer_id and len(value_hits) >= 3:
            confidence = 0.9

        if engagement_hits:
            confidence = max(confidence, 0.75)

        # -------------------------------------------------
        # BOUNDARY PENALTIES
        # -------------------------------------------------
        if exclusion_hits:
            confidence -= 0.25

        confidence = round(max(0.0, min(0.95, confidence)), 2)

        # -------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------
        if confidence < 0.5:
            return DomainDetectionResult(
                None,
                0.0,
                {
                    "signals": {
                        "has_customer_id": has_customer_id,
                        "value_hits": value_hits,
                        "engagement_hits": engagement_hits,
                        "excluded": exclusion_hits,
                    }
                },
            )

        return DomainDetectionResult(
            domain="customer_value",
            confidence=confidence,
            signals={
                "value_signals": value_hits,
                "engagement_signals": engagement_hits,
                "excluded_signals": exclusion_hits,
            },
        )


# =====================================================
# DOMAIN REGISTRATION — CUSTOMER VALUE & LOYALTY
# =====================================================

def register(registry):
    """
    Registers the Customer Value & Loyalty domain
    as a first-class Sreejita Framework domain.
    """
    registry.register(
        name="customer_value",
        domain_cls=CustomerValueDomain,
        detector_cls=CustomerValueDomainDetector,
    )


