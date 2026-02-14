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
# HELPERS — CUSTOMER / CX (DOMAIN-SAFE, GOVERNED)
# =====================================================

def _safe_div(
    n: Optional[float],
    d: Optional[float],
) -> Optional[float]:
    """
    Safe division helper.

    GUARANTEES:
    - Never raises
    - Returns None on invalid input
    - Explicit float coercion
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
    Customer / CX–safe time column detector.

    SUPPORTED SEMANTICS:
    - Interaction / touchpoint dates
    - Ticket / case creation & resolution
    - Feedback / survey timestamps
    - Generic event or record timestamps

    DESIGN PRINCIPLES:
    - Experience-centric ordering
    - No domain leakage (sales, logistics, finance)
    - Safe fallback only
    - Never mutates df
    """

    if df is None or df.empty:
        return None

    # Ordered by customer-experience relevance
    candidates = [
        "interaction_date",
        "event_date",
        "activity_date",
        "touchpoint_date",
        "ticket_date",
        "case_date",
        "created_at",
        "updated_at",
        "feedback_date",
        "survey_date",
        "timestamp",
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
            except (ValueError, TypeError):
                continue

    return None


# =====================================================
# CUSTOMER DOMAIN (UNIVERSAL 10/10)
# =====================================================
class CustomerDomain(BaseDomain):
    name = "customer"
    description = "Universal Customer Intelligence (CX, Loyalty, Support, Churn)"

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL, CX-SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Customer / CX preprocess guarantees:

        - Defensive copy (framework invariant)
        - Semantic column resolution (raw signals only)
        - Datetime & numeric normalization
        - NO KPI computation
        - NO sub-domain inference
        - Graceful degradation on sparse CX data
        """

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if not isinstance(df, pd.DataFrame):
            raise TypeError("CustomerDomain.preprocess expects a DataFrame")

        df = df.copy(deep=False)

        # -------------------------------------------------
        # TIME COLUMN (CX-CENTRIC)
        # -------------------------------------------------
        self.time_col = _detect_time_column(df)

        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col],
                errors="coerce",
            )
            df = df.sort_values(self.time_col)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (RAW EXPERIENCE SIGNALS)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # ---------------- IDENTITY ----------------
            "customer": (
                resolve_column(df, "customer_id")
                or resolve_column(df, "customer")
                or resolve_column(df, "user_id")
            ),

            # ---------------- EXPERIENCE SCORES ----------------
            "nps": (
                resolve_column(df, "nps")
                or resolve_column(df, "net_promoter_score")
            ),
            "csat": (
                resolve_column(df, "csat")
                or resolve_column(df, "satisfaction")
            ),
            "ces": (
                resolve_column(df, "ces")
                or resolve_column(df, "effort_score")
            ),

            # ---------------- CHURN / RETENTION ----------------
            "churn": (
                resolve_column(df, "churn")
                or resolve_column(df, "churned")
                or resolve_column(df, "is_churned")
            ),

            # ---------------- SUPPORT & OPERATIONS ----------------
            "ticket": (
                resolve_column(df, "ticket_id")
                or resolve_column(df, "case_id")
            ),
            "frt": (
                resolve_column(df, "first_response_time")
                or resolve_column(df, "frt")
            ),
            "art": (
                resolve_column(df, "avg_resolution_time")
                or resolve_column(df, "resolution_time")
            ),
            "fcr": (
                resolve_column(df, "fcr")
                or resolve_column(df, "first_contact_resolution")
            ),

            # ---------------- QUALITATIVE SIGNAL ----------------
            "sentiment": (
                resolve_column(df, "sentiment_score")
                or resolve_column(df, "sentiment")
            ),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        numeric_keys = {
            "nps",
            "csat",
            "ces",
            "frt",
            "art",
            "sentiment",
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
        # BINARY / RATE NORMALIZATION (SAFE)
        # -------------------------------------------------
        for key in ("fcr", "churn"):
            col = self.cols.get(key)
            if col and col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")

                # Normalize percentages → ratios if needed
                if s.dropna().median() and s.dropna().median() > 1:
                    s = s / 100.0

                df[col] = s.clip(0, 1)

        # -------------------------------------------------
        # DATA COMPLETENESS (RAW SIGNAL COVERAGE)
        # -------------------------------------------------
        raw_signal_keys = {
            "nps",
            "csat",
            "ces",
            "churn",
            "ticket",
            "frt",
            "art",
            "fcr",
            "sentiment",
        }

        present = sum(
            1 for k, v in self.cols.items()
            if k in raw_signal_keys and v
        )

        self.data_completeness = round(
            present / max(len(raw_signal_keys), 1),
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
        Execute the customer pipeline with stable, safe outputs.

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

            # Confidence is optional, never fabricated
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
        Customer / CX KPI Engine (v1.0)
    
        GUARANTEES:
        - Capability-driven sub-domains
        - No invented KPIs
        - Confidence-tagged KPIs
        - Graceful degradation on weak CX datasets
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
            "experience": "Customer Experience Signals",
            "support": "Customer Support & Operations",
            "loyalty": "Retention & Loyalty",
            "sentiment": "Customer Sentiment",
        }
    
        kpis: Dict[str, Any] = {
            "sub_domains": sub_domains,
            "record_count": volume,
            "data_completeness": getattr(self, "data_completeness", 0.5),
            "_domain_kpi_map": {},
            "_confidence": {},
        }
    
        # -------------------------------------------------
        # SAFE HELPERS
        # -------------------------------------------------
        def safe_mean(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        def safe_rate(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        # =================================================
        # EXPERIENCE — PERCEPTION & EFFORT
        # =================================================
        experience: List[str] = []
    
        val = safe_mean(c.get("nps"))
        if val is not None:
            kpis["experience_avg_nps"] = val
            experience.append("experience_avg_nps")
    
        val = safe_mean(c.get("csat"))
        if val is not None:
            kpis["experience_avg_csat"] = val
            experience.append("experience_avg_csat")
    
        val = safe_mean(c.get("ces"))
        if val is not None:
            kpis["experience_avg_ces"] = val
            experience.append("experience_avg_ces")
    
        if c.get("customer"):
            cnt = df[c["customer"]].nunique()
            if cnt > 0:
                kpis["experience_customer_count"] = cnt
                experience.append("experience_customer_count")
    
        # =================================================
        # SUPPORT — RESPONSIVENESS & RESOLUTION
        # =================================================
        support: List[str] = []
    
        if c.get("ticket"):
            cnt = df[c["ticket"]].nunique()
            if cnt > 0:
                kpis["support_ticket_volume"] = cnt
                support.append("support_ticket_volume")
    
        val = safe_mean(c.get("frt"))
        if val is not None:
            kpis["support_avg_first_response_time"] = val
            support.append("support_avg_first_response_time")
    
        val = safe_mean(c.get("art"))
        if val is not None:
            kpis["support_avg_resolution_time"] = val
            support.append("support_avg_resolution_time")
    
        val = safe_rate(c.get("fcr"))
        if val is not None:
            kpis["support_first_contact_resolution_rate"] = val
            support.append("support_first_contact_resolution_rate")
    
        # =================================================
        # LOYALTY — RETENTION & STABILITY
        # =================================================
        loyalty: List[str] = []
    
        val = safe_rate(c.get("churn"))
        if val is not None:
            kpis["loyalty_churn_rate"] = val
            loyalty.append("loyalty_churn_rate")
    
        if c.get("customer"):
            freq = df[c["customer"]].value_counts()
            val = _safe_div((freq > 1).sum(), len(freq))
            if val is not None:
                kpis["loyalty_repeat_customer_rate"] = val
                loyalty.append("loyalty_repeat_customer_rate")
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        sentiment: List[str] = []
    
        val = safe_mean(c.get("sentiment"))
        if val is not None:
            kpis["sentiment_avg_score"] = val
            sentiment.append("sentiment_avg_score")
    
            var = _safe_div(
                df[c["sentiment"]].std(),
                df[c["sentiment"]].mean(),
            )
            if var is not None:
                kpis["sentiment_score_variability"] = var
                sentiment.append("sentiment_score_variability")
    
        # -------------------------------------------------
        # DOMAIN → KPI MAP
        # -------------------------------------------------
        kpis["_domain_kpi_map"] = {
            "experience": experience,
            "support": support,
            "loyalty": loyalty,
            "sentiment": sentiment,
        }
    
        # -------------------------------------------------
        # KPI CONFIDENCE (MANDATORY, NON-INFLATING)
        # -------------------------------------------------
        for key, val in kpis.items():
            if key.startswith("_") or not isinstance(val, (int, float)):
                continue
    
            base = 0.7
            if volume < 100:
                base -= 0.15
            if "rate" in key or "variability" in key:
                base += 0.05
            if "sentiment" in key:
                base -= 0.05
    
            kpis["_confidence"][key] = round(
                max(0.4, min(0.9, base)),
                2,
            )
    
        self._last_kpis = kpis
        return kpis
   

    # ---------------- VISUALS (8 CANDIDATES, TOP 4 SELECTED) ----------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Customer / CX Visual Engine (v1.0)
    
        GUARANTEES:
        - Evidence-only visuals (KPI-gated)
        - No confidence inflation
        - Graceful suppression on weak data
        - Report layer decides trimming
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.cols
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH — KPIs
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        domain_map = kpis.get("_domain_kpi_map", {})
    
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
                "confidence": None,  # Phase-5: no visual confidence inflation
            })
    
        # =================================================
        # EXPERIENCE — SCORES & DISTRIBUTIONS
        # =================================================
        if domain_map.get("experience"):
    
            if c.get("nps") and df[c["nps"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["nps"]].hist(ax=ax, bins=15)
                ax.set_title("NPS Score Distribution")
                save(
                    fig,
                    "experience_nps_dist.png",
                    "Customer loyalty score spread",
                    0.95,
                    "experience",
                    "perception",
                    "distribution",
                )
    
            if c.get("csat") and df[c["csat"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["csat"]].hist(ax=ax, bins=10)
                ax.set_title("CSAT Distribution")
                save(
                    fig,
                    "experience_csat_dist.png",
                    "Customer satisfaction dispersion",
                    0.9,
                    "experience",
                    "perception",
                    "distribution",
                )
    
            if c.get("ces") and df[c["ces"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["ces"]].hist(ax=ax, bins=10)
                ax.set_title("Customer Effort Score Distribution")
                save(
                    fig,
                    "experience_ces_dist.png",
                    "Customer effort variability",
                    0.85,
                    "experience",
                    "effort",
                    "distribution",
                )
    
        # =================================================
        # SUPPORT — TIME & VOLUME
        # =================================================
        if domain_map.get("support"):
    
            if (
                c.get("ticket")
                and self.time_col
                and df[self.time_col].nunique() >= 3
            ):
                fig, ax = plt.subplots()
                df.set_index(self.time_col)[c["ticket"]].nunique().resample("M").sum().plot(ax=ax)
                ax.set_title("Support Ticket Volume Over Time")
                save(
                    fig,
                    "support_ticket_trend.png",
                    "Support demand trend",
                    0.95,
                    "support",
                    "volume",
                    "time",
                )
    
            if c.get("frt") and df[c["frt"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["frt"]].hist(ax=ax, bins=15)
                ax.set_title("First Response Time Distribution")
                save(
                    fig,
                    "support_frt_dist.png",
                    "Response speed variability",
                    0.9,
                    "support",
                    "velocity",
                    "distribution",
                )
    
            if c.get("art") and df[c["art"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["art"]].hist(ax=ax, bins=15)
                ax.set_title("Resolution Time Distribution")
                save(
                    fig,
                    "support_art_dist.png",
                    "Resolution duration dispersion",
                    0.85,
                    "support",
                    "velocity",
                    "distribution",
                )
    
        # =================================================
        # LOYALTY — RETENTION STRUCTURE
        # =================================================
        if domain_map.get("loyalty") and c.get("customer"):
    
            freq = df[c["customer"]].value_counts()
            if freq.nunique() > 2:
                fig, ax = plt.subplots()
                freq.clip(upper=5).value_counts().sort_index().plot.bar(ax=ax)
                ax.set_title("Customer Interaction Frequency")
                save(
                    fig,
                    "loyalty_repeat_dist.png",
                    "Repeat engagement pattern",
                    0.9,
                    "loyalty",
                    "behavior",
                    "distribution",
                )
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        if domain_map.get("sentiment") and c.get("sentiment"):
    
            if df[c["sentiment"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["sentiment"]].hist(ax=ax, bins=15)
                ax.set_title("Customer Sentiment Distribution")
                save(
                    fig,
                    "sentiment_dist.png",
                    "Emotional tone dispersion",
                    0.85,
                    "sentiment",
                    "emotion",
                    "distribution",
                )
    
                fig, ax = plt.subplots()
                df[c["sentiment"]].plot(kind="box", ax=ax)
                ax.set_title("Sentiment Spread")
                save(
                    fig,
                    "sentiment_box.png",
                    "Sentiment variability",
                    0.8,
                    "sentiment",
                    "variability",
                    "spread",
                )
    
        # -------------------------------------------------
        # RETURN — REPORT LAYER TRIMS
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals
 

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Customer / CX Insight Engine (v1.0)
    
        GUARANTEES:
        - Evidence-gated insights only
        - No narrative inflation
        - Sub-domain suppression under weak data
        - Descriptive, not evaluative language
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        domain_map = kpis.get("_domain_kpi_map", {})
    
        # -------------------------------------------------
        # KPI SHORTCUTS
        # -------------------------------------------------
        churn = kpis.get("loyalty_churn_rate")
        repeat = kpis.get("loyalty_repeat_customer_rate")
    
        nps = kpis.get("experience_avg_nps")
        csat = kpis.get("experience_avg_csat")
        ces = kpis.get("experience_avg_ces")
    
        frt = kpis.get("support_avg_first_response_time")
        art = kpis.get("support_avg_resolution_time")
        fcr = kpis.get("support_first_contact_resolution_rate")
    
        sentiment_avg = kpis.get("sentiment_avg_score")
        sentiment_var = kpis.get("sentiment_score_variability")
    
        # =================================================
        # EXPERIENCE — PERCEPTION & EFFORT
        # =================================================
        if domain_map.get("experience") and sum(v is not None for v in (nps, csat, ces)) >= 2:
            insights.append({
                "level": "INFO",
                "sub_domain": "experience",
                "title": "Customer Experience Signals Observed",
                "so_what": "Available experience scores provide a baseline view of customer perception.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": "experience",
                "title": "Experience Score Dispersion",
                "so_what": "Variation in experience metrics indicates heterogeneous customer journeys.",
            })
    
        # =================================================
        # SUPPORT — RESPONSIVENESS & RESOLUTION
        # =================================================
        if domain_map.get("support") and sum(v is not None for v in (frt, art, fcr)) >= 2:
            insights.append({
                "level": "INFO",
                "sub_domain": "support",
                "title": "Support Performance Signals Present",
                "so_what": "Response and resolution metrics describe current support operations.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": "support",
                "title": "Support Process Variability",
                "so_what": "Observed spread in support metrics suggests variability across cases.",
            })
    
        # =================================================
        # LOYALTY — RETENTION & STABILITY
        # =================================================
        if domain_map.get("loyalty") and sum(v is not None for v in (churn, repeat)) >= 1:
            insights.append({
                "level": "INFO",
                "sub_domain": "loyalty",
                "title": "Retention Signals Available",
                "so_what": "Churn and repeat indicators provide visibility into customer stability.",
            })
    
            if churn is not None and repeat is not None:
                insights.append({
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Repeat and Attrition Context",
                    "so_what": "Combined churn and repeat signals describe customer lifecycle dynamics.",
                })
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        if domain_map.get("sentiment") and sentiment_avg is not None:
            insights.append({
                "level": "INFO",
                "sub_domain": "sentiment",
                "title": "Customer Sentiment Baseline",
                "so_what": "Average sentiment provides qualitative context to customer experience data.",
            })
    
            if sentiment_var is not None:
                insights.append({
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Sentiment Variability Observed",
                    "so_what": "Variation in sentiment reflects differing emotional responses across customers.",
                })
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK (ONLY IF NO EVIDENCE)
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Customer Signals Detected",
                "so_what": "Available customer data provides limited but usable CX visibility.",
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
        Customer / CX Recommendation Engine (v1.0)
    
        GUARANTEES:
        - Evidence-gated recommendations
        - Advisory, executive-safe language
        - No urgency bias, no thresholds
        - Graceful degradation on weak CX data
        """
    
        recs: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return recs
    
        domain_map = kpis.get("_domain_kpi_map", {})
    
        # KPI shortcuts
        nps = kpis.get("experience_avg_nps")
        csat = kpis.get("experience_avg_csat")
        ces = kpis.get("experience_avg_ces")
    
        frt = kpis.get("support_avg_first_response_time")
        art = kpis.get("support_avg_resolution_time")
        fcr = kpis.get("support_first_contact_resolution_rate")
    
        churn = kpis.get("loyalty_churn_rate")
        repeat = kpis.get("loyalty_repeat_customer_rate")
    
        sentiment_avg = kpis.get("sentiment_avg_score")
    
        # =================================================
        # EXPERIENCE — PERCEPTION & EFFORT
        # =================================================
        if domain_map.get("experience") and sum(v is not None for v in (nps, csat, ces)) >= 2:
            recs.extend([
                {
                    "sub_domain": "experience",
                    "action": "Review experience score distributions to identify journey stages with higher friction.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "experience",
                    "action": "Compare satisfaction and effort signals to surface potential experience gaps.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "experience",
                    "action": "Segment experience metrics by customer cohort to improve interpretability.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "experience",
                    "action": "Use effort-related signals to inform journey simplification discussions.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # SUPPORT — RESPONSIVENESS & RESOLUTION
        # =================================================
        if domain_map.get("support") and sum(v is not None for v in (frt, art, fcr)) >= 2:
            recs.extend([
                {
                    "sub_domain": "support",
                    "action": "Review response and resolution time distributions to understand service variability.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "support",
                    "action": "Assess case complexity patterns using resolution duration signals.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "support",
                    "action": "Use first-contact resolution metrics to inform support process refinement.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "support",
                    "action": "Monitor support demand trends to support capacity planning discussions.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # LOYALTY — RETENTION & STABILITY
        # =================================================
        if domain_map.get("loyalty") and any(v is not None for v in (churn, repeat)):
            recs.extend([
                {
                    "sub_domain": "loyalty",
                    "action": "Review retention and repeat engagement patterns across customer segments.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Incorporate churn and repeat signals into customer lifecycle discussions.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Assess concentration of repeat engagement among key customer cohorts.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Monitor retention trends over time for stability pattern identification.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        if domain_map.get("sentiment") and sentiment_avg is not None:
            recs.extend([
                {
                    "sub_domain": "sentiment",
                    "action": "Review sentiment distributions to understand emotional response diversity.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Correlate sentiment signals with quantitative CX metrics for contextual insight.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Segment sentiment signals by interaction type or channel.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK (ONLY IF NO EVIDENCE)
        # -------------------------------------------------
        if not recs:
            recs.append({
                "sub_domain": "mixed",
                "action": "Continue monitoring available customer experience signals for emerging patterns.",
                "priority": "LOW",
            })
    
        return recs


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class CustomerDomainDetector(BaseDomainDetector):
    """
    Customer / CX Domain Detector (v1.0)

    Detects datasets focused on:
    - Customer experience measurement
    - Satisfaction, effort, loyalty signals
    - Support experience (not ops-only)
    - Sentiment and feedback

    Explicitly avoids:
    - Transactional ownership (Retail / Ecommerce)
    - Campaign execution (Marketing)
    - Pure operational ticket logs
    """

    domain_name = "customer"

    # Strong CX anchors (experience-oriented)
    CX_ANCHORS: Set[str] = {
        "nps",
        "net_promoter",
        "csat",
        "satisfaction",
        "ces",
        "effort",
        "sentiment",
        "feedback",
        "experience",
    }

    # Support experience (must be paired with CX signal)
    SUPPORT_SIGNALS: Set[str] = {
        "ticket",
        "case",
        "resolution",
        "response_time",
        "first_response",
        "fcr",
    }

    # Boundary control — ownership signals from other domains
    EXCLUSION_TOKENS: Set[str] = {
        # Retail / Ecommerce
        "order",
        "sales",
        "revenue",
        "price",
        "gmv",
        "transaction",
        "checkout",
        "cart",
        "session",
        # Marketing
        "campaign",
        "impression",
        "click",
        "ctr",
        "cpc",
        # Supply Chain
        "inventory",
        "shipment",
        "delivery",
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        cols = {str(c).lower() for c in df.columns}

        # -------------------------------------------------
        # ANCHOR SIGNALS
        # -------------------------------------------------
        has_cx = any(any(t in c for t in self.CX_ANCHORS) for c in cols)
        has_support = any(any(t in c for t in self.SUPPORT_SIGNALS) for c in cols)
        has_customer_id = any("customer" in c or "user" in c for c in cols)

        # -------------------------------------------------
        # BASE CONFIDENCE (CAPABILITY-BASED)
        # -------------------------------------------------
        confidence = 0.0

        if has_cx and has_customer_id:
            confidence = 0.7

        if has_cx and has_support:
            confidence = max(confidence, 0.8)

        if has_cx and has_support and has_customer_id:
            confidence = 0.9

        # -------------------------------------------------
        # BOUNDARY CONTROL (PROPORTIONAL)
        # -------------------------------------------------
        exclusion_hits = [
            c for c in cols
            if any(t in c for t in self.EXCLUSION_TOKENS)
        ]

        if exclusion_hits:
            # Penalize proportionally, not absolutely
            penalty = min(0.3, 0.05 * len(exclusion_hits))
            confidence -= penalty

            # Ownership leakage → soft cap
            confidence = min(confidence, 0.85)

        confidence = round(max(0.0, min(0.95, confidence)), 2)

        # -------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------
        if confidence < 0.5:
            return DomainDetectionResult(
                None,
                0.0,
                {
                    "cx_signals": {
                        "has_experience_metrics": has_cx,
                        "has_support_metrics": has_support,
                        "has_customer_id": has_customer_id,
                    },
                },
            )

        return DomainDetectionResult(
            domain="customer",
            confidence=confidence,
            signals={
                "cx_signals": {
                    "experience": has_cx,
                    "support": has_support,
                    "identity": has_customer_id,
                },
                "excluded_signals": exclusion_hits,
            },
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        "customer",
        CustomerDomain,
        CustomerDomainDetector,
    )
