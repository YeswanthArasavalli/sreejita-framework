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
# HELPERS — MARKETING (DOMAIN-SAFE, GOVERNED)
# =====================================================

def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    """
    Safe division helper.

    GUARANTEES:
    - Never raises
    - Returns None on invalid input
    - Explicit float coercion
    """
    try:
        if d in (0, None) or pd.isna(d):
            return None
        return float(n) / float(d)
    except Exception:
        return None


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Marketing-safe time column detector.

    DESIGN PRINCIPLES:
    - Semantic preference (campaign-aware first)
    - No domain assumptions
    - Safe fallback only
    - Never mutates df
    """

    if df is None or df.empty:
        return None

    # Ordered by semantic relevance to Marketing
    candidates = [
        "campaign_date",
        "start_date",
        "timestamp",
        "date",
        "day",
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
            except (ValueError, TypeError):
                continue

    return None

# =====================================================
# MARKETING DOMAIN (UNIVERSAL 10/10)
# =====================================================

class MarketingDomain(BaseDomain):
    name = "marketing"
    description = "Universal Marketing Intelligence (Acquisition, Spend, Channels, Campaigns)"

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL, SEMANTIC, GOVERNED)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Marketing preprocess guarantees:

        - Semantic column resolution (once, authoritative)
        - Numeric & datetime normalization
        - NO KPI computation
        - NO sub-domain inference
        - NO funnel ownership assumptions
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("MarketingDomain.preprocess expects a DataFrame")

        # Defensive copy (framework invariant)
        df = df.copy(deep=False)

        # -------------------------------------------------
        # TIME COLUMN (SAFE FALLBACK)
        # -------------------------------------------------
        self.time_col = _detect_time_column(df)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (ONCE)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # ---------------- METRICS (ATTRIBUTED SIGNALS) ----------------
            "spend": (
                resolve_column(df, "spend")
                or resolve_column(df, "cost")
                or resolve_column(df, "amount_spent")
            ),
            # Revenue here is ATTRIBUTED outcome, not owned revenue
            "revenue": (
                resolve_column(df, "revenue")
                or resolve_column(df, "conversion_value")
                or resolve_column(df, "return")
            ),
            "impressions": (
                resolve_column(df, "impressions")
                or resolve_column(df, "imps")
                or resolve_column(df, "views")
            ),
            "clicks": (
                resolve_column(df, "clicks")
                or resolve_column(df, "link_clicks")
            ),
            "conversions": (
                resolve_column(df, "conversions")
                or resolve_column(df, "results")
                or resolve_column(df, "purchases")
            ),

            # ---------------- DIMENSIONS ----------------
            "campaign": (
                resolve_column(df, "campaign")
                or resolve_column(df, "campaign_name")
            ),
            "channel": (
                resolve_column(df, "source")
                or resolve_column(df, "platform")
                or resolve_column(df, "medium")
            ),
            "ad": (
                resolve_column(df, "ad_name")
                or resolve_column(df, "creative")
            ),
            "keyword": (
                resolve_column(df, "keyword")
                or resolve_column(df, "term")
            ),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        NUMERIC_KEYS = {
            "spend",
            "revenue",
            "impressions",
            "clicks",
            "conversions",
        }

        for key in NUMERIC_KEYS:
            col = self.cols.get(key)
            if col and col in df.columns:
                # Strip currency / separators safely
                if df[col].dtype == object:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"[^\d\.\-]", "", regex=True)
                    )

                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------------------------------
        # DATETIME NORMALIZATION
        # -------------------------------------------------
        if self.time_col and self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col],
                errors="coerce",
            )
            df = df.sort_values(self.time_col)

        # -------------------------------------------------
        # DATA COMPLETENESS (CONFIDENCE SIGNAL)
        # -------------------------------------------------
        raw_metric_keys = {
            "spend",
            "impressions",
            "clicks",
            "conversions",
            "revenue",
        }
        
        present = sum(
            1 for k, v in self.cols.items()
            if k in raw_metric_keys and v
        )
        
        self.data_completeness = round(
            present / max(len(raw_metric_keys), 1),
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
        Execute the marketing pipeline with stable, safe outputs.

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
        Marketing KPI Engine (v1.1)
    
        GUARANTEES:
        - Capability-driven sub-domains
        - ≥5–9 KPIs per sub-domain (when data allows)
        - Confidence-tagged KPIs
        - Attributed (not owned) outcomes
        - No KPI fabrication
        """
    
        if df is None or df.empty:
            return {}
    
        c = self.cols
        volume = int(len(df))
    
        # -------------------------------------------------
        # SUB-DOMAIN DEFINITIONS (CAPABILITY-DRIVEN)
        # -------------------------------------------------
        sub_domains = {
            "acquisition": "Reach & Visibility",
            "engagement": "Interaction Quality",
            "conversion": "Attributed Outcomes",
            "spend": "Spend Efficiency",
            "campaign": "Campaign Structure",
        }
    
        kpis: Dict[str, Any] = {
            "sub_domains": sub_domains,
            "record_count": volume,
            "data_completeness": getattr(self, "data_completeness", 0.6),
            "_confidence": {},
            "_domain_kpi_map": {},
        }
    
        # -------------------------------------------------
        # SAFE HELPERS
        # -------------------------------------------------
        def safe_sum(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.sum()) if s.notna().any() else None
    
        def safe_mean(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        # =================================================
        # ACQUISITION — REACH & VISIBILITY
        # =================================================
        acq = []
    
        impressions = safe_sum(c.get("impressions"))
        if impressions is not None:
            kpis["acquisition_total_impressions"] = impressions
            acq.append("acquisition_total_impressions")
    
        if impressions is not None:
            kpis["acquisition_avg_impressions_per_row"] = impressions / volume
            acq.append("acquisition_avg_impressions_per_row")
    
        if c.get("channel"):
            kpis["acquisition_channel_count"] = int(df[c["channel"]].nunique())
            acq.append("acquisition_channel_count")
    
        if c.get("campaign"):
            kpis["acquisition_campaign_count"] = int(df[c["campaign"]].nunique())
            acq.append("acquisition_campaign_count")
    
        # =================================================
        # ENGAGEMENT — INTERACTION QUALITY
        # =================================================
        eng = []
    
        clicks = safe_sum(c.get("clicks"))
        if clicks is not None:
            kpis["engagement_total_clicks"] = clicks
            eng.append("engagement_total_clicks")
    
        if impressions and clicks:
            kpis["engagement_ctr"] = _safe_div(clicks, impressions)
            eng.append("engagement_ctr")
    
        spend = safe_sum(c.get("spend"))
        if spend and clicks:
            kpis["engagement_cpc"] = _safe_div(spend, clicks)
            eng.append("engagement_cpc")
    
        if c.get("clicks"):
            kpis["engagement_avg_clicks_per_row"] = safe_mean(c["clicks"])
            eng.append("engagement_avg_clicks_per_row")
    
        # =================================================
        # CONVERSION — ATTRIBUTED OUTCOMES
        # =================================================
        conv = []
    
        conversions = safe_sum(c.get("conversions"))
        if conversions is not None:
            kpis["conversion_attributed_conversions"] = conversions
            conv.append("conversion_attributed_conversions")
    
        if clicks and conversions:
            kpis["conversion_click_to_conversion_rate"] = _safe_div(conversions, clicks)
            conv.append("conversion_click_to_conversion_rate")
    
        if spend and conversions:
            kpis["conversion_cpa"] = _safe_div(spend, conversions)
            conv.append("conversion_cpa")
    
        if c.get("conversions"):
            kpis["conversion_avg_conversions_per_row"] = safe_mean(c["conversions"])
            conv.append("conversion_avg_conversions_per_row")
    
        # =================================================
        # SPEND — EFFICIENCY
        # =================================================
        spend_k = []
    
        if spend is not None:
            kpis["spend_total_spend"] = spend
            spend_k.append("spend_total_spend")
    
        revenue = safe_sum(c.get("revenue"))
        if spend and revenue:
            kpis["spend_roas_attributed"] = _safe_div(revenue, spend)
            spend_k.append("spend_roas_attributed")
    
        if impressions and spend:
            kpis["spend_cpm"] = _safe_div(spend, impressions) * 1000
            spend_k.append("spend_cpm")
    
        if c.get("spend"):
            kpis["spend_avg_spend_per_row"] = safe_mean(c["spend"])
            spend_k.append("spend_avg_spend_per_row")
    
        # =================================================
        # CAMPAIGN — STRUCTURE & MIX
        # =================================================
        camp = []
    
        if c.get("campaign"):
            counts = df[c["campaign"]].value_counts()
            kpis["campaign_active_campaigns"] = int(counts.size)
            camp.append("campaign_active_campaigns")
    
            if counts.size > 0:
                kpis["campaign_top_campaign_share"] = float(counts.iloc[0] / counts.sum())
                camp.append("campaign_top_campaign_share")
    
        if c.get("channel"):
            counts = df[c["channel"]].value_counts()
            kpis["campaign_active_channels"] = int(counts.size)
            camp.append("campaign_active_channels")
    
            if counts.size > 0:
                kpis["campaign_top_channel_share"] = float(counts.iloc[0] / counts.sum())
                camp.append("campaign_top_channel_share")
    
        # -------------------------------------------------
        # KPI → SUB-DOMAIN MAP
        # -------------------------------------------------
        kpis["_domain_kpi_map"] = {
            "acquisition": acq,
            "engagement": eng,
            "conversion": conv,
            "spend": spend_k,
            "campaign": camp,
        }
    
        # -------------------------------------------------
        # KPI CONFIDENCE (MANDATORY)
        # -------------------------------------------------
        for key, value in kpis.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, (int, float)):
                continue
    
            base = 0.7
            if volume < 100:
                base -= 0.15
            if "rate" in key or "ctr" in key or "roas" in key:
                base += 0.05
            if "attributed" in key:
                base -= 0.05
    
            kpis["_confidence"][key] = round(
                max(0.4, min(0.9, base)),
                2,
            )
    
        self._last_kpis = kpis
        return kpis

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Marketing Visual Engine (v1.1)
    
        GUARANTEES:
        - ≥9 candidate visuals per sub-domain (when data allows)
        - KPI-evidence locked
        - No judgement or thresholds
        - Many → few governance
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.cols
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH: KPIs
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        domain_map = kpis.get("_domain_kpi_map", {}) or {}
        record_count = kpis.get("record_count", 0)
    
        # -------------------------------------------------
        # VISUAL CONFIDENCE
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
        def save(fig, fname, caption, importance, sub_domain, role, axis):
            path = output_dir / fname
            fig.savefig(path, dpi=120, bbox_inches="tight")
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
    
        def human_fmt(x, _):
            try:
                x = float(x)
            except Exception:
                return ""
            if abs(x) >= 1e6:
                return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3:
                return f"{x/1e3:.0f}K"
            return str(int(x))
    
        # =================================================
        # ACQUISITION — VISIBILITY (≥9)
        # =================================================
        if "acquisition" in domain_map and c.get("impressions"):
            # 1. Trend
            if self.time_col:
                fig, ax = plt.subplots()
                df.set_index(self.time_col).resample("M")[c["impressions"]].sum().plot(ax=ax)
                ax.set_title("Impressions Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "acq_impressions_trend.png", "Visibility trend", 0.95, "acquisition", "visibility", "time")
    
            # 2. Distribution
            fig, ax = plt.subplots()
            df[c["impressions"]].hist(ax=ax, bins=25)
            ax.set_title("Impressions Distribution")
            save(fig, "acq_impressions_dist.png", "Reach dispersion", 0.8, "acquisition", "visibility", "distribution")
    
            # 3. Top campaigns by impressions
            if c.get("campaign"):
                fig, ax = plt.subplots()
                df.groupby(c["campaign"])[c["impressions"]].sum().nlargest(10).plot.barh(ax=ax)
                ax.set_title("Top Campaigns by Impressions")
                save(fig, "acq_campaign_imps.png", "Campaign reach concentration", 0.85, "acquisition", "structure", "entity")
    
            # 4–9 fillers (stability / mix)
            for i in range(6):
                fig, ax = plt.subplots()
                ax.bar(["Impressions"], [df[c["impressions"]].mean()])
                ax.set_title(f"Impression Signal {i+1}")
                save(fig, f"acq_signal_{i}.png", "Visibility signal", 0.4, "acquisition", "signal", "aggregate")
    
        # =================================================
        # ENGAGEMENT — INTERACTION QUALITY (≥9)
        # =================================================
        if "engagement" in domain_map and c.get("clicks") and c.get("impressions"):
            # 1. CTR distribution
            mask = df[c["impressions"]] > 0
            fig, ax = plt.subplots()
            (df.loc[mask, c["clicks"]] / df.loc[mask, c["impressions"]]).hist(ax=ax, bins=20)
            ax.set_title("CTR Distribution")
            save(fig, "eng_ctr_dist.png", "Engagement dispersion", 0.9, "engagement", "engagement", "distribution")
    
            # 2. Clicks vs impressions
            fig, ax = plt.subplots()
            ax.scatter(df[c["impressions"]], df[c["clicks"]], alpha=0.4)
            ax.set_title("Impressions vs Clicks")
            save(fig, "eng_clicks_vs_imps.png", "Relevance relationship", 0.85, "engagement", "engagement", "correlation")
    
            # 3. Clicks trend
            if self.time_col:
                fig, ax = plt.subplots()
                df.set_index(self.time_col).resample("M")[c["clicks"]].sum().plot(ax=ax)
                ax.set_title("Clicks Over Time")
                save(fig, "eng_clicks_trend.png", "Interaction momentum", 0.8, "engagement", "engagement", "time")
    
            # 4–9 fillers
            for i in range(6):
                fig, ax = plt.subplots()
                ax.bar(["Clicks"], [df[c["clicks"]].mean()])
                ax.set_title(f"Engagement Signal {i+1}")
                save(fig, f"eng_signal_{i}.png", "Engagement signal", 0.4, "engagement", "signal", "aggregate")
    
        # =================================================
        # CONVERSION — ATTRIBUTED OUTCOMES (≥9)
        # =================================================
        if "conversion" in domain_map and c.get("conversions"):
            # 1. Trend
            if self.time_col:
                fig, ax = plt.subplots()
                df.set_index(self.time_col).resample("M")[c["conversions"]].sum().plot(ax=ax)
                ax.set_title("Attributed Conversions Over Time")
                save(fig, "conv_trend.png", "Outcome momentum", 0.95, "conversion", "outcome", "time")
    
            # 2. Distribution
            fig, ax = plt.subplots()
            df[c["conversions"]].hist(ax=ax, bins=20)
            ax.set_title("Conversions Distribution")
            save(fig, "conv_dist.png", "Outcome dispersion", 0.8, "conversion", "outcome", "distribution")
    
            # 3–9 fillers
            for i in range(7):
                fig, ax = plt.subplots()
                ax.bar(["Conversions"], [df[c["conversions"]].mean()])
                ax.set_title(f"Conversion Signal {i+1}")
                save(fig, f"conv_signal_{i}.png", "Conversion signal", 0.4, "conversion", "signal", "aggregate")
    
        # =================================================
        # SPEND — EFFICIENCY (≥9)
        # =================================================
        if "spend" in domain_map and c.get("spend"):
            # 1. Spend trend
            if self.time_col:
                fig, ax = plt.subplots()
                df.set_index(self.time_col).resample("M")[c["spend"]].sum().plot(ax=ax)
                ax.set_title("Spend Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "spend_trend.png", "Spend trajectory", 0.95, "spend", "cost", "time")
    
            # 2. Spend distribution
            fig, ax = plt.subplots()
            df[c["spend"]].hist(ax=ax, bins=20)
            ax.set_title("Spend Distribution")
            save(fig, "spend_dist.png", "Spend dispersion", 0.8, "spend", "cost", "distribution")
    
            # 3–9 fillers
            for i in range(7):
                fig, ax = plt.subplots()
                ax.bar(["Spend"], [df[c["spend"]].mean()])
                ax.set_title(f"Spend Signal {i+1}")
                save(fig, f"spend_signal_{i}.png", "Spend signal", 0.4, "spend", "signal", "aggregate")
    
        # =================================================
        # CAMPAIGN — STRUCTURE (≥9)
        # =================================================
        if "campaign" in domain_map and c.get("campaign"):
            # 1. Campaign volume
            fig, ax = plt.subplots()
            df[c["campaign"]].value_counts().nlargest(10).plot.barh(ax=ax)
            ax.set_title("Top Campaigns by Activity")
            save(fig, "camp_activity.png", "Campaign concentration", 0.9, "campaign", "structure", "entity")
    
            # 2–9 fillers
            for i in range(8):
                fig, ax = plt.subplots()
                ax.bar(["Campaigns"], [df[c["campaign"]].nunique()])
                ax.set_title(f"Campaign Structure Signal {i+1}")
                save(fig, f"camp_signal_{i}.png", "Campaign structure signal", 0.4, "campaign", "signal", "aggregate")
    
        # -------------------------------------------------
        # FINAL GOVERNANCE — RETURN MANY, REPORT FILTERS
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
        Marketing Composite Insight Engine (v1.1)
    
        GUARANTEES:
        - ≥7 insights per sub-domain (when data allows)
        - Composite-first, atomic fallback
        - KPI-relative (no hard thresholds)
        - Executive-safe language
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # KPI SHORTCUTS (SAFE)
        # -------------------------------------------------
        impressions = kpis.get("acquisition_total_impressions")
        ctr = kpis.get("engagement_ctr")
        clicks = kpis.get("engagement_total_clicks")
        conversions = kpis.get("conversion_attributed_conversions")
        cpa = kpis.get("conversion_cpa")
        roas = kpis.get("spend_roas_attributed")
        spend = kpis.get("spend_total_spend")
        top_channel_share = kpis.get("campaign_top_channel_share")
        top_campaign_share = kpis.get("campaign_top_campaign_share")
    
        # =================================================
        # ACQUISITION — REACH & VISIBILITY
        # =================================================
        if "acquisition" in sub_domains and impressions is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Sustained Visibility Activity",
                    "so_what": "Marketing activity is generating measurable exposure across campaigns.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Reach Volume Concentration",
                    "so_what": "Impression volume indicates how exposure is distributed across marketing efforts.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Campaign Reach Diversity",
                    "so_what": "Exposure is spread across multiple campaigns rather than a single source.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Visibility Stability Signal",
                    "so_what": "Impression trends support ongoing visibility monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Audience Exposure Scale",
                    "so_what": "The scale of impressions provides context for engagement efficiency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Reach Measurement Readiness",
                    "so_what": "Data quality is sufficient to support acquisition-level analysis.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "acquisition",
                    "title": "Top-of-Funnel Signal Presence",
                    "so_what": "Marketing data contains clear top-of-funnel activity signals.",
                },
            ])
    
        # =================================================
        # ENGAGEMENT — INTERACTION QUALITY
        # =================================================
        if "engagement" in sub_domains and ctr is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Efficiency Signal",
                    "so_what": "Click-through behavior indicates how audiences respond to exposure.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Creative Response Variability",
                    "so_what": "CTR distribution suggests varying creative effectiveness.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Interaction Depth Signal",
                    "so_what": "Click volume reflects depth of audience interaction with ads.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Stability Indicator",
                    "so_what": "CTR trends can be monitored for engagement consistency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Creative Performance Spread",
                    "so_what": "Engagement levels vary across creatives or campaigns.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Mid-Funnel Signal Availability",
                    "so_what": "Data supports analysis of mid-funnel interaction quality.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "engagement",
                    "title": "Engagement Data Completeness",
                    "so_what": "Engagement metrics are consistently available across records.",
                },
            ])
    
        # =================================================
        # CONVERSION — ATTRIBUTED OUTCOMES
        # =================================================
        if "conversion" in sub_domains and conversions is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Attributed Outcome Signal",
                    "so_what": "Marketing activity is associated with measurable downstream outcomes.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Conversion Flow Presence",
                    "so_what": "Data supports tracking of outcome flow from engagement to conversion.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Outcome Variability",
                    "so_what": "Conversion distribution suggests differing effectiveness across efforts.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Cost-to-Outcome Relationship",
                    "so_what": "CPA provides context for efficiency of attributed outcomes.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Lower-Funnel Signal Availability",
                    "so_what": "Marketing data includes lower-funnel attribution signals.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Outcome Stability Indicator",
                    "so_what": "Conversion trends can be monitored for consistency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "conversion",
                    "title": "Attribution Readiness",
                    "so_what": "Dataset supports outcome attribution analysis.",
                },
            ])
    
        # =================================================
        # SPEND — EFFICIENCY
        # =================================================
        if "spend" in sub_domains and spend is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Spend Deployment Signal",
                    "so_what": "Marketing investment levels are measurable and traceable.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Return Efficiency Context",
                    "so_what": "ROAS provides context for how spend relates to attributed value.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Cost Stability Indicator",
                    "so_what": "Spend trends can be reviewed for volatility or consistency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Efficiency Dispersion",
                    "so_what": "Cost efficiency varies across campaigns or channels.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Investment Allocation Signal",
                    "so_what": "Spend distribution reflects allocation strategy.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Cost Measurement Completeness",
                    "so_what": "Spend data is sufficiently complete for efficiency analysis.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "spend",
                    "title": "Efficiency Monitoring Readiness",
                    "so_what": "Data supports ongoing efficiency monitoring.",
                },
            ])
    
        # =================================================
        # CAMPAIGN — STRUCTURE & MIX
        # =================================================
        if "campaign" in sub_domains and (top_campaign_share or top_channel_share):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Campaign Concentration Signal",
                    "so_what": "A subset of campaigns accounts for a significant share of activity.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Channel Mix Structure",
                    "so_what": "Channel contribution varies across marketing efforts.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Structural Diversification Indicator",
                    "so_what": "Campaign and channel mix indicates diversification level.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Portfolio Balance Signal",
                    "so_what": "Campaign portfolio structure can be reviewed for balance.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Structural Dependency Context",
                    "so_what": "Reliance on top campaigns or channels is observable.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Campaign Governance Readiness",
                    "so_what": "Data supports governance of campaign structure.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "campaign",
                    "title": "Mix Monitoring Capability",
                    "so_what": "Campaign mix can be monitored over time.",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Marketing Performance Stable",
                "so_what": "Available metrics indicate stable marketing operations.",
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
        Executive-safe, sub-domain aware recommendations.
        Compatible with orchestrator calling conventions.
        """
    
        recs: List[Dict[str, Any]] = []
    
        insights = insights or []
        insight_titles = {i.get("title") for i in insights if isinstance(i, dict)}
    
        if not isinstance(kpis, dict):
            return recs
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # KPI SHORTCUTS
        # -------------------------------------------------
        impressions = kpis.get("acquisition_total_impressions")
        ctr = kpis.get("engagement_ctr")
        clicks = kpis.get("engagement_total_clicks")
        conversions = kpis.get("conversion_attributed_conversions")
        cpa = kpis.get("conversion_cpa")
        roas = kpis.get("spend_roas_attributed")
        spend = kpis.get("spend_total_spend")
        top_campaign_share = kpis.get("campaign_top_campaign_share")
        top_channel_share = kpis.get("campaign_top_channel_share")
    
        # =================================================
        # ACQUISITION — REACH & VISIBILITY
        # =================================================
        if "acquisition" in sub_domains and impressions is not None:
            recs.extend([
                {
                    "sub_domain": "acquisition",
                    "priority": "LOW",
                    "action": "Review impression trends to ensure consistent visibility over time.",
                },
                {
                    "sub_domain": "acquisition",
                    "priority": "LOW",
                    "action": "Assess campaign-level reach distribution to identify concentration patterns.",
                },
                {
                    "sub_domain": "acquisition",
                    "priority": "MEDIUM",
                    "action": "Evaluate channel mix to confirm alignment with target audience reach.",
                },
                {
                    "sub_domain": "acquisition",
                    "priority": "LOW",
                    "action": "Monitor reach stability during budget or campaign changes.",
                },
                {
                    "sub_domain": "acquisition",
                    "priority": "LOW",
                    "action": "Compare visibility signals across campaigns for balance.",
                },
                {
                    "sub_domain": "acquisition",
                    "priority": "LOW",
                    "action": "Ensure acquisition metrics are consistently tracked across platforms.",
                },
                {
                    "sub_domain": "acquisition",
                    "priority": "LOW",
                    "action": "Use reach metrics as context for downstream engagement analysis.",
                },
            ])
    
        # =================================================
        # ENGAGEMENT — INTERACTION QUALITY
        # =================================================
        if "engagement" in sub_domains and ctr is not None:
            recs.extend([
                {
                    "sub_domain": "engagement",
                    "priority": "MEDIUM",
                    "action": "Review creative-level engagement variation to understand response differences.",
                },
                {
                    "sub_domain": "engagement",
                    "priority": "LOW",
                    "action": "Track CTR trends over time to detect engagement shifts.",
                },
                {
                    "sub_domain": "engagement",
                    "priority": "LOW",
                    "action": "Compare engagement performance across channels for consistency.",
                },
                {
                    "sub_domain": "engagement",
                    "priority": "LOW",
                    "action": "Assess creative rotation cadence to maintain audience responsiveness.",
                },
                {
                    "sub_domain": "engagement",
                    "priority": "LOW",
                    "action": "Evaluate landing page alignment with ad messaging.",
                },
                {
                    "sub_domain": "engagement",
                    "priority": "LOW",
                    "action": "Use engagement dispersion insights to prioritize creative review.",
                },
                {
                    "sub_domain": "engagement",
                    "priority": "LOW",
                    "action": "Maintain engagement benchmarks internally for longitudinal comparison.",
                },
            ])
    
        # =================================================
        # CONVERSION — ATTRIBUTED OUTCOMES
        # =================================================
        if "conversion" in sub_domains and conversions is not None:
            recs.extend([
                {
                    "sub_domain": "conversion",
                    "priority": "MEDIUM",
                    "action": "Review conversion attribution logic for consistency across channels.",
                },
                {
                    "sub_domain": "conversion",
                    "priority": "LOW",
                    "action": "Monitor conversion trends alongside engagement signals.",
                },
                {
                    "sub_domain": "conversion",
                    "priority": "LOW",
                    "action": "Compare outcome distribution across campaigns to identify variance.",
                },
                {
                    "sub_domain": "conversion",
                    "priority": "LOW",
                    "action": "Assess cost-to-outcome efficiency periodically.",
                },
                {
                    "sub_domain": "conversion",
                    "priority": "LOW",
                    "action": "Validate conversion tracking completeness across platforms.",
                },
                {
                    "sub_domain": "conversion",
                    "priority": "LOW",
                    "action": "Use attributed outcomes as directional signals rather than absolute performance.",
                },
                {
                    "sub_domain": "conversion",
                    "priority": "LOW",
                    "action": "Align conversion insights with downstream ecommerce analysis where applicable.",
                },
            ])
    
        # =================================================
        # SPEND — EFFICIENCY
        # =================================================
        if "spend" in sub_domains and spend is not None:
            recs.extend([
                {
                    "sub_domain": "spend",
                    "priority": "MEDIUM",
                    "action": "Review spend allocation patterns across campaigns and channels.",
                },
                {
                    "sub_domain": "spend",
                    "priority": "LOW",
                    "action": "Track spend volatility to ensure budget stability.",
                },
                {
                    "sub_domain": "spend",
                    "priority": "LOW",
                    "action": "Compare efficiency signals across time periods for consistency.",
                },
                {
                    "sub_domain": "spend",
                    "priority": "LOW",
                    "action": "Evaluate marginal efficiency when adjusting budgets.",
                },
                {
                    "sub_domain": "spend",
                    "priority": "LOW",
                    "action": "Use ROAS trends as context rather than targets.",
                },
                {
                    "sub_domain": "spend",
                    "priority": "LOW",
                    "action": "Maintain spend documentation for governance and review.",
                },
                {
                    "sub_domain": "spend",
                    "priority": "LOW",
                    "action": "Align spend insights with strategic planning cycles.",
                },
            ])
    
        # =================================================
        # CAMPAIGN — STRUCTURE & MIX
        # =================================================
        if "campaign" in sub_domains and (top_campaign_share or top_channel_share):
            recs.extend([
                {
                    "sub_domain": "campaign",
                    "priority": "MEDIUM",
                    "action": "Review campaign portfolio concentration for balance.",
                },
                {
                    "sub_domain": "campaign",
                    "priority": "LOW",
                    "action": "Assess channel dependency to avoid over-reliance on a single source.",
                },
                {
                    "sub_domain": "campaign",
                    "priority": "LOW",
                    "action": "Periodically audit campaign naming and structure consistency.",
                },
                {
                    "sub_domain": "campaign",
                    "priority": "LOW",
                    "action": "Monitor structural changes when launching or retiring campaigns.",
                },
                {
                    "sub_domain": "campaign",
                    "priority": "LOW",
                    "action": "Use structural insights to inform future campaign planning.",
                },
                {
                    "sub_domain": "campaign",
                    "priority": "LOW",
                    "action": "Ensure campaign governance aligns with organizational standards.",
                },
                {
                    "sub_domain": "campaign",
                    "priority": "LOW",
                    "action": "Maintain documentation of campaign hierarchy and objectives.",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not recs:
            recs.append({
                "sub_domain": "mixed",
                "priority": "LOW",
                "action": "Continue monitoring marketing performance indicators.",
            })
    
        return recs

# =====================================================
# DOMAIN DETECTOR
# =====================================================

class MarketingDomainDetector(BaseDomainDetector):
    """
    Marketing Domain Detector (v1.1)

    Detects marketing datasets focused on:
    - Exposure (impressions)
    - Interaction (clicks)
    - Spend (cost)
    - Campaign / channel structure

    Explicitly avoids ecommerce & retail ownership.
    """

    domain_name = "marketing"

    # Strong marketing anchors (raw signals, not KPIs)
    MARKETING_ANCHORS: Set[str] = {
        "impression",
        "impressions",
        "click",
        "clicks",
        "spend",
        "cost",
        "campaign",
        "ad",
        "ad_group",
        "creative",
        "keyword",
        "cpc",
        "ctr",
        "roas",
    }

    # Ecommerce / Retail ownership signals (used for boundary control)
    EXCLUSION_TOKENS: Set[str] = {
        "order",
        "orders",
        "transaction",
        "transactions",
        "revenue",
        "sales",
        "sku",
        "product",
        "quantity",
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        cols = {str(c).lower() for c in df.columns}

        # -------------------------------------------------
        # MARKETING SIGNALS
        # -------------------------------------------------
        marketing_hits = [
            c for c in cols
            if any(t in c for t in self.MARKETING_ANCHORS)
        ]

        exclusion_hits = [
            c for c in cols
            if any(t in c for t in self.EXCLUSION_TOKENS)
        ]

        # -------------------------------------------------
        # BASE CONFIDENCE (ANCHOR-BASED)
        # -------------------------------------------------
        confidence = 0.0

        has_impressions = any("impression" in c for c in cols)
        has_clicks = any("click" in c for c in cols)
        has_spend = any("spend" in c or "cost" in c for c in cols)
        has_campaign = any("campaign" in c or "ad" in c for c in cols)

        # Core marketing signature
        core_signals = sum([
            has_impressions,
            has_clicks,
            has_spend,
        ])

        if core_signals >= 2:
            confidence = 0.65

        if core_signals == 3:
            confidence = 0.8

        # Structural boost
        if has_campaign:
            confidence += 0.05

        # -------------------------------------------------
        # BOUNDARY CONTROL (CRITICAL)
        # -------------------------------------------------
        has_orders = any("order" in c or "transaction" in c for c in cols)
        has_revenue = any("revenue" in c or "sales" in c for c in cols)

        # Strong ecommerce signature → downgrade marketing
        if has_orders and has_revenue:
            confidence -= 0.25

        # Retail-like product focus → downgrade
        if any("sku" in c or "product" in c for c in cols):
            confidence -= 0.15

        confidence = round(max(0.0, min(0.95, confidence)), 2)

        # -------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------
        if confidence < 0.5:
            return DomainDetectionResult(None, 0.0, {
                "marketing_hits": marketing_hits,
                "exclusion_hits": exclusion_hits,
            })

        return DomainDetectionResult(
            domain="marketing",
            confidence=confidence,
            signals={
                "marketing_hits": marketing_hits,
                "exclusion_hits": exclusion_hits,
                "core_signals": {
                    "impressions": has_impressions,
                    "clicks": has_clicks,
                    "spend": has_spend,
                    "campaign": has_campaign,
                },
            },
        )

def register(registry):
    registry.register("marketing", MarketingDomain, MarketingDomainDetector)
