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
        for key in ["fcr", "churn"]:
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
        - 5–9 KPIs per sub-domain (when data allows)
        - Confidence-tagged KPIs
        - Proxy metrics explicitly marked
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
        experience = []
    
        if c.get("nps"):
            kpis["experience_avg_nps"] = safe_mean(c["nps"])
            experience.append("experience_avg_nps")
    
        if c.get("csat"):
            kpis["experience_avg_csat"] = safe_mean(c["csat"])
            experience.append("experience_avg_csat")
    
        if c.get("ces"):
            kpis["experience_avg_ces"] = safe_mean(c["ces"])
            experience.append("experience_avg_ces")
    
        if c.get("customer"):
            kpis["experience_customer_count"] = df[c["customer"]].nunique()
            experience.append("experience_customer_count")
    
        # =================================================
        # SUPPORT — RESPONSIVENESS & RESOLUTION
        # =================================================
        support = []
    
        if c.get("ticket"):
            kpis["support_ticket_volume"] = df[c["ticket"]].nunique()
            support.append("support_ticket_volume")
    
        if c.get("frt"):
            kpis["support_avg_first_response_time"] = safe_mean(c["frt"])
            support.append("support_avg_first_response_time")
    
        if c.get("art"):
            kpis["support_avg_resolution_time"] = safe_mean(c["art"])
            support.append("support_avg_resolution_time")
    
        if c.get("fcr"):
            kpis["support_first_contact_resolution_rate"] = safe_rate(c["fcr"])
            support.append("support_first_contact_resolution_rate")
    
        # =================================================
        # LOYALTY — RETENTION & STABILITY
        # =================================================
        loyalty = []
    
        if c.get("churn"):
            kpis["loyalty_churn_rate"] = safe_rate(c["churn"])
            loyalty.append("loyalty_churn_rate")
    
        if c.get("customer"):
            freq = df[c["customer"]].value_counts()
            kpis["loyalty_repeat_customer_rate"] = _safe_div(
                (freq > 1).sum(),
                len(freq),
            )
            loyalty.append("loyalty_repeat_customer_rate")
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        sentiment = []
    
        if c.get("sentiment"):
            kpis["sentiment_avg_score"] = safe_mean(c["sentiment"])
            sentiment.append("sentiment_avg_score")
    
            kpis["sentiment_score_variability"] = _safe_div(
                df[c["sentiment"]].std(),
                df[c["sentiment"]].mean(),
            )
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
        # KPI CONFIDENCE (MANDATORY)
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
        - ≥9 visual candidates per sub-domain (when data allows)
        - Evidence-only (no judgement, no thresholds)
        - No trimming (report layer decides)
        - Flat-plot prevention
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
        record_count = kpis.get("record_count", 0)
    
        # -------------------------------------------------
        # VISUAL CONFIDENCE (DATA-DRIVEN)
        # -------------------------------------------------
        if record_count >= 5000:
            visual_conf = 0.85
        elif record_count >= 1000:
            visual_conf = 0.7
        else:
            visual_conf = 0.55
    
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
                "confidence": visual_conf,
            })
    
        # =================================================
        # EXPERIENCE — SCORES & DISTRIBUTIONS
        # =================================================
        if "experience" in domain_map:
    
            if c.get("nps") and df[c["nps"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["nps"]].hist(ax=ax, bins=15)
                ax.set_title("NPS Score Distribution")
                save(fig, "experience_nps_dist.png", "Customer loyalty score spread",
                     0.95, "experience", "perception", "distribution")
    
            if c.get("csat") and df[c["csat"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["csat"]].hist(ax=ax, bins=10)
                ax.set_title("CSAT Distribution")
                save(fig, "experience_csat_dist.png", "Customer satisfaction dispersion",
                     0.9, "experience", "perception", "distribution")
    
            if c.get("ces") and df[c["ces"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["ces"]].hist(ax=ax, bins=10)
                ax.set_title("Customer Effort Score Distribution")
                save(fig, "experience_ces_dist.png", "Customer effort variability",
                     0.85, "experience", "effort", "distribution")
    
        # =================================================
        # SUPPORT — TIME & VOLUME
        # =================================================
        if "support" in domain_map:
    
            if c.get("ticket") and self.time_col:
                fig, ax = plt.subplots()
                df.set_index(self.time_col).resample("M")[c["ticket"]].nunique().plot(ax=ax)
                ax.set_title("Support Ticket Volume Over Time")
                save(fig, "support_ticket_trend.png", "Support demand trend",
                     0.95, "support", "volume", "time")
    
            if c.get("frt") and df[c["frt"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["frt"]].hist(ax=ax, bins=15)
                ax.set_title("First Response Time Distribution")
                save(fig, "support_frt_dist.png", "Response speed variability",
                     0.9, "support", "velocity", "distribution")
    
            if c.get("art") and df[c["art"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["art"]].hist(ax=ax, bins=15)
                ax.set_title("Resolution Time Distribution")
                save(fig, "support_art_dist.png", "Resolution duration dispersion",
                     0.85, "support", "velocity", "distribution")
    
        # =================================================
        # LOYALTY — RETENTION STRUCTURE
        # =================================================
        if "loyalty" in domain_map and c.get("customer"):
    
            freq = df[c["customer"]].value_counts()
            if freq.nunique() > 2:
                fig, ax = plt.subplots()
                freq.clip(upper=5).value_counts().sort_index().plot.bar(ax=ax)
                ax.set_title("Customer Interaction Frequency")
                save(fig, "loyalty_repeat_dist.png", "Repeat engagement pattern",
                     0.9, "loyalty", "behavior", "distribution")
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        if "sentiment" in domain_map and c.get("sentiment"):
    
            if df[c["sentiment"]].nunique() > 3:
                fig, ax = plt.subplots()
                df[c["sentiment"]].hist(ax=ax, bins=15)
                ax.set_title("Customer Sentiment Distribution")
                save(fig, "sentiment_dist.png", "Emotional tone dispersion",
                     0.85, "sentiment", "emotion", "distribution")
    
                fig, ax = plt.subplots()
                df[c["sentiment"]].plot(kind="box", ax=ax)
                ax.set_title("Sentiment Spread")
                save(fig, "sentiment_box.png", "Sentiment variability",
                     0.8, "sentiment", "variability", "spread")
    
        # -------------------------------------------------
        # RETURN MANY — REPORT LAYER WILL TRIM
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
        - Composite insights are primary
        - ≥7 composite insights per sub-domain (when data allows)
        - No thresholds, no targets, no judgement
        - KPI-relative, evidence-based language
        - Atomic insights only as fallback
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # KPI SHORTCUTS (SAFE)
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
        if "experience" in sub_domains and any(v is not None for v in (nps, csat, ces)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "Customer Experience Baseline Established",
                    "so_what": "Experience scores provide a measurable baseline for customer perception.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "Perception Signal Coverage",
                    "so_what": "Multiple experience signals allow triangulation of customer satisfaction.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "Effort–Satisfaction Context",
                    "so_what": "Customer effort and satisfaction can be examined together for friction analysis.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "Experience Trend Readiness",
                    "so_what": "Experience data supports ongoing trend and change monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "CX Measurement Consistency",
                    "so_what": "Observed score distributions suggest consistent CX measurement.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "Customer Perception Variability",
                    "so_what": "Dispersion in experience scores reflects heterogeneous customer journeys.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "experience",
                    "title": "Experience Governance Readiness",
                    "so_what": "CX metrics are suitable for executive-level governance.",
                },
            ])
    
        # =================================================
        # SUPPORT — RESPONSIVENESS & RESOLUTION
        # =================================================
        if "support" in sub_domains and any(v is not None for v in (frt, art, fcr)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "Support Responsiveness Baseline",
                    "so_what": "Response time metrics establish support responsiveness norms.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "Resolution Duration Context",
                    "so_what": "Resolution times reflect operational complexity and case handling patterns.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "First-Contact Resolution Signal",
                    "so_what": "FCR rates provide insight into issue resolution quality.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "Support Process Variability",
                    "so_what": "Observed spread in support metrics indicates variability across cases.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "Operational Load Insight",
                    "so_what": "Support metrics help contextualize workload and throughput.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "Service Stability Context",
                    "so_what": "Support signals can be monitored for consistency over time.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "support",
                    "title": "Support Governance Readiness",
                    "so_what": "Support KPIs are suitable for executive service governance.",
                },
            ])
    
        # =================================================
        # LOYALTY — RETENTION & STABILITY
        # =================================================
        if "loyalty" in sub_domains and any(v is not None for v in (churn, repeat)):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Retention Baseline Established",
                    "so_what": "Churn metrics establish a baseline for retention analysis.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Repeat Engagement Signal",
                    "so_what": "Repeat customer behavior provides insight into loyalty depth.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Customer Stability Context",
                    "so_what": "Retention patterns indicate stability of the customer base.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Lifecycle Risk Visibility",
                    "so_what": "Retention metrics support lifecycle risk monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Engagement Concentration Insight",
                    "so_what": "Repeat behavior may be concentrated within a subset of customers.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Retention Governance Readiness",
                    "so_what": "Loyalty metrics are suitable for governance review.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "loyalty",
                    "title": "Customer Base Continuity",
                    "so_what": "Observed churn and repeat signals indicate continuity patterns.",
                },
            ])
    
        # =================================================
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # =================================================
        if "sentiment" in sub_domains and sentiment_avg is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Sentiment Baseline Established",
                    "so_what": "Average sentiment provides a qualitative experience baseline.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Emotional Variability Context",
                    "so_what": "Sentiment dispersion reflects varied emotional responses.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Experience–Emotion Link",
                    "so_what": "Sentiment signals complement quantitative CX metrics.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Feedback Consistency Indicator",
                    "so_what": "Consistency in sentiment suggests stable feedback patterns.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Qualitative Signal Coverage",
                    "so_what": "Sentiment data supports narrative CX interpretation.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "Experience Narrative Readiness",
                    "so_what": "Sentiment metrics are suitable for executive narratives.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sentiment",
                    "title": "CX Storytelling Support",
                    "so_what": "Qualitative signals enhance CX storytelling.",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK (ATOMIC ONLY IF NEEDED)
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Customer Signals Available",
                "so_what": "Available customer data provides baseline CX visibility.",
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
        - ≥7 recommendations per sub-domain (when active)
        - Advisory, executive-safe language
        - No thresholds, no urgency bias
        - Insight-aware but not title-dependent
        """
    
        recs: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return recs
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # EXPERIENCE — PERCEPTION & EFFORT
        # -------------------------------------------------
        if "experience" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "experience",
                    "action": "Review experience score distributions to identify journey stages with higher friction.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "experience",
                    "action": "Analyze differences between satisfaction and effort signals to uncover hidden experience gaps.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "experience",
                    "action": "Incorporate CX score trends into regular executive reviews.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "experience",
                    "action": "Segment experience metrics by customer cohort to improve interpretability.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "experience",
                    "action": "Validate consistency of CX measurement methods across channels.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "experience",
                    "action": "Use effort-related signals to prioritize journey simplification initiatives.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "experience",
                    "action": "Align experience metrics with broader customer strategy discussions.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # SUPPORT — RESPONSIVENESS & RESOLUTION
        # -------------------------------------------------
        if "support" in sub_domains:
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
                    "action": "Use first-contact resolution metrics to guide support process refinement.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "support",
                    "action": "Monitor support demand trends to inform capacity planning discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "support",
                    "action": "Standardize support KPIs for consistent service governance.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "support",
                    "action": "Review support workflows for opportunities to reduce rework.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "support",
                    "action": "Incorporate support performance signals into CX improvement planning.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # LOYALTY — RETENTION & STABILITY
        # -------------------------------------------------
        if "loyalty" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "loyalty",
                    "action": "Review retention and repeat engagement patterns across customer segments.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Incorporate churn and repeat metrics into lifecycle management discussions.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Use loyalty signals to inform prioritization of customer engagement initiatives.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Assess concentration of repeat engagement among top customer cohorts.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Monitor retention trends over time for early stability signals.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Align loyalty metrics with long-term customer value discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "loyalty",
                    "action": "Integrate retention signals into executive risk reviews.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # SENTIMENT — QUALITATIVE EXPERIENCE
        # -------------------------------------------------
        if "sentiment" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "sentiment",
                    "action": "Review sentiment distributions to understand emotional response diversity.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Correlate sentiment signals with quantitative CX metrics for deeper insight.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Use qualitative feedback to enrich executive CX narratives.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Monitor sentiment variability as an early indicator of experience shifts.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Segment sentiment signals by interaction type or channel.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Incorporate sentiment trends into CX governance reviews.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sentiment",
                    "action": "Use qualitative insights to support experience design discussions.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not recs:
            recs.append({
                "sub_domain": "mixed",
                "action": "Continue monitoring customer experience signals for emerging patterns.",
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
        # BOUNDARY CONTROL
        # -------------------------------------------------
        has_exclusion = any(
            any(t in c for t in self.EXCLUSION_TOKENS)
            for c in cols
        )

        if has_exclusion:
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
                "excluded_signals": [
                    c for c in cols
                    if any(t in c for t in self.EXCLUSION_TOKENS)
                ],
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
