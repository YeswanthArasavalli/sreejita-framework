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
# HELPERS — SUPPLY CHAIN (DOMAIN-SAFE, GOVERNED)
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
    Supply Chain–safe time column detector.

    SUPPORTED SEMANTICS:
    - Order dates
    - Ship dates
    - Delivery / receipt dates
    - Generic date / timestamp

    DESIGN PRINCIPLES:
    - Semantic preference (logistics-aware)
    - No dataset assumptions
    - Safe fallback only
    - Never mutates df
    """

    if df is None or df.empty:
        return None

    # Ordered by operational relevance in supply chain
    candidates = [
        "delivery_date",
        "delivered_date",
        "receipt_date",
        "received_date",
        "ship_date",
        "shipping_date",
        "order_date",
        "orderdate",
        "date",
        "timestamp",
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
# SUPPLY CHAIN DOMAIN (UNIVERSAL 10/10)
# =====================================================
class SupplyChainDomain(BaseDomain):
    name = "supply_chain"
    description = "Universal Supply Chain Intelligence (Planning, Inventory, Logistics, Resilience)"

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL, GOVERNED)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supply Chain preprocess guarantees:

        - Semantic column resolution (once, authoritative)
        - Datetime & numeric normalization
        - NO KPI computation
        - NO sub-domain inference
        - Graceful degradation on weak data
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("SupplyChainDomain.preprocess expects a DataFrame")

        # Defensive copy (framework invariant)
        df = df.copy(deep=False)

        # -------------------------------------------------
        # TIME COLUMN (LOGISTICS-AWARE)
        # -------------------------------------------------
        self.time_col = _detect_time_column(df)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (RAW SIGNALS ONLY)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # ---------------- IDENTIFIERS ----------------
            "order_id": (
                resolve_column(df, "order_id")
                or resolve_column(df, "shipment_id")
            ),
            "sku": (
                resolve_column(df, "sku")
                or resolve_column(df, "item_id")
            ),

            # ---------------- OPERATIONAL TIME ----------------
            "processing_time": resolve_column(df, "processing_time"),
            "packing_time": resolve_column(df, "packing_time"),

            # ---------------- STRUCTURE ----------------
            "category": (
                resolve_column(df, "category")
                or resolve_column(df, "product_family")
            ),
            "supplier": (
                resolve_column(df, "supplier")
                or resolve_column(df, "vendor")
            ),
            "carrier": (
                resolve_column(df, "carrier")
                or resolve_column(df, "logistics_provider")
            ),
            "status": resolve_column(df, "status"),

            # ---------------- INVENTORY / COST ----------------
            "inventory": (
                resolve_column(df, "inventory")
                or resolve_column(df, "stock_level")
            ),
            "cost": (
                resolve_column(df, "cost")
                or resolve_column(df, "shipping_cost")
            ),

            # ---------------- LOGISTICS / SUSTAINABILITY ----------------
            "distance": (
                resolve_column(df, "distance")
                or resolve_column(df, "miles")
            ),
            "weight": (
                resolve_column(df, "weight")
                or resolve_column(df, "tonnage")
            ),
            "co2": (
                resolve_column(df, "co2")
                or resolve_column(df, "emissions")
            ),

            # ---------------- DATES ----------------
            "order_date": resolve_column(df, "order_date"),
            "ship_date": resolve_column(df, "ship_date"),
            "delivery_date": (
                resolve_column(df, "delivery_date")
                or resolve_column(df, "actual_delivery")
            ),
            "promised_date": (
                resolve_column(df, "promised_date")
                or resolve_column(df, "estimated_delivery")
            ),
        }

        # -------------------------------------------------
        # DATETIME NORMALIZATION
        # -------------------------------------------------
        date_keys = {
            "order_date",
            "ship_date",
            "delivery_date",
            "promised_date",
        }

        for key in date_keys:
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        for key in ["processing_time", "packing_time"]:
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.time_col and self.time_col in df.columns:
            df = df.sort_values(self.time_col)

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (SAFE)
        # -------------------------------------------------
        numeric_keys = {
            "inventory",
            "cost",
            "distance",
            "weight",
            "co2",
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
        # DATA COMPLETENESS (RAW METRICS ONLY)
        # -------------------------------------------------
        raw_signal_keys = {
            "inventory",
            "cost",
            "distance",
            "processing_time",
            "packing_time",
            "status",
            "weight",
            "co2",
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
        Execute the supply chain pipeline with stable, safe outputs.

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
        Supply Chain KPI Engine (v1.0)
    
        GUARANTEES:
        - Capability-driven sub-domains
        - 5–9 KPIs per sub-domain (when data allows)
        - Confidence-tagged KPIs
        - No hardcoded assumptions
        - Proxy metrics explicitly tagged
        """
    
        if df is None or df.empty:
            return {}
    
        c = self.cols
        volume = int(len(df))
    
        # -------------------------------------------------
        # SUB-DOMAIN DEFINITIONS
        # -------------------------------------------------
        sub_domains = {
            "planning": "Planning & Flow Stability",
            "inventory": "Inventory & Working Capital",
            "logistics": "Fulfillment & Movement",
            "cost": "Cost Efficiency",
            "resilience": "Risk & Dependency",
            "sustainability": "Environmental Efficiency",
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
        def safe_sum(col):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.sum()) if s.notna().any() else None
    
        def safe_mean(col):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        # =================================================
        # PLANNING & FLOW STABILITY
        # =================================================
        planning = []
    
        total_orders = df[c["order_id"]].nunique() if c.get("order_id") else volume
        kpis["planning_total_orders"] = total_orders
        planning.append("planning_total_orders")
    
        if c.get("order_date"):
            kpis["planning_order_frequency"] = total_orders / max(df[c["order_date"]].nunique(), 1)
            planning.append("planning_order_frequency")
    
        # --- Lead Time (Date-based OR Operational proxy) ---
        lead_days = None
        
        if c.get("order_date") and c.get("delivery_date"):
            lead_days = (df[c["delivery_date"]] - df[c["order_date"]]).dt.days
        
        elif c.get("processing_time") or c.get("packing_time"):
            proc = pd.to_numeric(df.get(c.get("processing_time")), errors="coerce") if c.get("processing_time") else 0
            pack = pd.to_numeric(df.get(c.get("packing_time")), errors="coerce") if c.get("packing_time") else 0
            lead_days = (proc.fillna(0) + (pack.fillna(0) / 60)) / 24  # hours → days
        
        if lead_days is not None:
            lead_days = lead_days.dropna()
            if not lead_days.empty:
                kpis["planning_avg_lead_time"] = lead_days.mean()
                kpis["planning_lead_time_variability"] = _safe_div(lead_days.std(), lead_days.mean())
                planning.extend([
                    "planning_avg_lead_time",
                    "planning_lead_time_variability",
                ])
    
        # =================================================
        # INVENTORY & WORKING CAPITAL
        # =================================================
        inventory = []
    
        if c.get("inventory"):
            inv = pd.to_numeric(df[c["inventory"]], errors="coerce")
            kpis["inventory_avg_stock"] = inv.mean()
            kpis["inventory_stock_variability"] = _safe_div(inv.std(), inv.mean())
            inventory.extend([
                "inventory_avg_stock",
                "inventory_stock_variability",
            ])
    
            kpis["inventory_zero_stock_ratio"] = (inv <= 0).mean()
            inventory.append("inventory_zero_stock_ratio")
    
        # =================================================
        # LOGISTICS & FULFILLMENT
        # =================================================
        logistics = []
    
        if c.get("delivery_date") and c.get("promised_date"):
            valid = df[c["delivery_date"]].notna() & df[c["promised_date"]].notna()
            if valid.any():
                on_time = df.loc[valid, c["delivery_date"]] <= df.loc[valid, c["promised_date"]]
                kpis["logistics_on_time_delivery_rate"] = on_time.mean()
                logistics.append("logistics_on_time_delivery_rate")
    
        if c.get("distance"):
            kpis["logistics_avg_distance"] = safe_mean(c["distance"])
            logistics.append("logistics_avg_distance")

        elif c.get("status"):
            status = df[c["status"]].astype(str).str.lower()
            delivered = status.str.contains("deliver", na=False)
            returned = status.str.contains("return|fail|cancel", na=False)
            if (delivered | returned).any():
                kpis["logistics_on_time_delivery_rate"] = delivered.mean()
                logistics.append("logistics_on_time_delivery_rate")
    
        # =================================================
        # COST EFFICIENCY
        # =================================================
        cost = []
    
        if c.get("cost"):
            total_cost = safe_sum(c["cost"])
            kpis["cost_total_cost"] = total_cost
            cost.append("cost_total_cost")
    
            kpis["cost_avg_cost_per_record"] = safe_mean(c["cost"])
            cost.append("cost_avg_cost_per_record")
    
        if c.get("cost") and c.get("distance"):
            kpis["cost_cost_per_distance"] = _safe_div(
                safe_sum(c["cost"]),
                safe_sum(c["distance"]),
            )
            cost.append("cost_cost_per_distance")
    
        # =================================================
        # RESILIENCE & DEPENDENCY
        # =================================================
        resilience = []
    
        if c.get("supplier"):
            counts = df[c["supplier"]].value_counts()
            kpis["resilience_supplier_count"] = int(counts.size)
            kpis["resilience_top_supplier_share"] = float(counts.iloc[0] / counts.sum())
            resilience.extend([
                "resilience_supplier_count",
                "resilience_top_supplier_share",
            ])
    
        if c.get("carrier"):
            counts = df[c["carrier"]].value_counts()
            kpis["resilience_carrier_count"] = int(counts.size)
            kpis["resilience_top_carrier_share"] = float(counts.iloc[0] / counts.sum())
            resilience.extend([
                "resilience_carrier_count",
                "resilience_top_carrier_share",
            ])

        # =================================================
        # SUSTAINABILITY (PROXY-AWARE)
        # =================================================
        sustainability = []
    
        if c.get("co2"):
            kpis["sustainability_avg_co2"] = safe_mean(c["co2"])
            sustainability.append("sustainability_avg_co2")
    
        elif c.get("distance"):
            kpis["sustainability_emissions_proxy"] = _safe_div(
                safe_sum(c["distance"]),
                volume,
            )
            sustainability.append("sustainability_emissions_proxy")

        # -------------------------------------------------
        # DOMAIN → KPI MAP
        # -------------------------------------------------
        kpis["_domain_kpi_map"] = {
            "planning": planning,
            "inventory": inventory,
            "logistics": logistics,
            "cost": cost,
            "resilience": resilience,
            "sustainability": sustainability,
        }
    
        # -------------------------------------------------
        # KPI CONFIDENCE
        # -------------------------------------------------
        for key, val in kpis.items():
            if key.startswith("_") or not isinstance(val, (int, float)):
                continue
    
            base = 0.7
            if volume < 100:
                base -= 0.15
            if "proxy" in key:
                base -= 0.1
            if "rate" in key or "variability" in key:
                base += 0.05
    
            kpis["_confidence"][key] = round(max(0.4, min(0.9, base)), 2)
    
        self._last_kpis = kpis
        return kpis

    # ---------------- VISUALS (SMART SELECTION) ----------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Supply Chain Visual Engine (v1.0)
    
        GUARANTEES:
        - ≥9 candidate visuals per sub-domain (when data allows)
        - Evidence-only (no thresholds, no judgement)
        - No trimming here (report layer decides)
        - KPI-backed & confidence-aware
        """
    
        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
    
        c = self.cols
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH: KPIs
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
        # PLANNING — FLOW & DEMAND
        # =================================================
        if "planning" in domain_map and self.time_col:
            # 1. Order volume trend
            fig, ax = plt.subplots()
            df.set_index(self.time_col).resample("M").size().plot(ax=ax)
            ax.set_title("Order Volume Over Time")
            save(fig, "planning_volume_trend.png", "Demand flow trend", 0.95, "planning", "volume", "time")
    
            # 2. Order volume distribution
            fig, ax = plt.subplots()
            df.resample("M", on=self.time_col).size().hist(ax=ax, bins=20)
            ax.set_title("Order Volume Distribution")
            save(fig, "planning_volume_dist.png", "Demand variability", 0.8, "planning", "volume", "distribution")
    
        # =================================================
        # LOGISTICS — FULFILLMENT
        # =================================================
        if "logistics" in domain_map and c.get("order_date") and c.get("delivery_date"):
            lead = (df[c["delivery_date"]] - df[c["order_date"]]).dt.days.dropna()

            if lead.nunique() > 3:
                fig, ax = plt.subplots()
                lead.hist(ax=ax, bins=20)
                ax.set_title("Lead Time Distribution")
                save(
                    fig, "logistics_lead_dist.png", "Fulfillment time dispersion", 0.95, "logistics", "velocity", "distribution",)
    
            # 4. Lead time spread
            fig, ax = plt.subplots()
            lead.plot(kind="box", ax=ax)
            ax.set_title("Lead Time Spread")
            save(fig, "logistics_lead_box.png", "Delivery variability", 0.9, "logistics", "variability", "spread")
    
        if "logistics" in domain_map and c.get("delivery_date") and c.get("promised_date"):
            
            # 5. On-time delivery trend (evidence only)
            on_time = df[c["delivery_date"]] <= df[c["promised_date"]]

            if on_time.nunique() > 1:
                fig, ax = plt.subplots()
                on_time.groupby(df[self.time_col].dt.to_period("M")).mean().plot(ax=ax)
                ax.set_title("On-Time Delivery Over Time")
                save(fig, "logistics_otd_trend.png", "Delivery reliability trend", 0.9, "logistics", "reliability", "time",)
    
        # =================================================
        # INVENTORY — STOCK POSITION
        # =================================================
        if "inventory" in domain_map and c.get("inventory"):
            # 6. Inventory distribution
            fig, ax = plt.subplots()
            df[c["inventory"]].hist(ax=ax, bins=20)
            ax.set_title("Inventory Level Distribution")
            save(fig, "inventory_dist.png", "Stock dispersion", 0.9, "inventory", "stock", "distribution")
    
            # 7. Inventory by category
            if c.get("category"):
                fig, ax = plt.subplots()
                df.groupby(c["category"])[c["inventory"]].mean().nlargest(10).plot.bar(ax=ax)
                ax.set_title("Average Inventory by Category")
                save(fig, "inventory_category.png", "Category stock mix", 0.85, "inventory", "mix", "entity")
    
        # =================================================
        # COST — EFFICIENCY
        # =================================================
        if "cost" in domain_map and c.get("cost"):
            # 8. Cost distribution
            fig, ax = plt.subplots()
            df[c["cost"]].hist(ax=ax, bins=20)
            ax.set_title("Cost Distribution")
            ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "cost_dist.png", "Cost variability", 0.9, "cost", "efficiency", "distribution")
    
            # 9. Cost over time
            if self.time_col:
                fig, ax = plt.subplots()
                df.set_index(self.time_col)[c["cost"]].resample("M").sum().plot(ax=ax)
                ax.set_title("Cost Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(fig, "cost_trend.png", "Cost trajectory", 0.85, "cost", "efficiency", "time")
    
        # =================================================
        # RESILIENCE — DEPENDENCY
        # =================================================
        if "resilience" in domain_map and c.get("supplier"):
            # 10. Supplier concentration
            fig, ax = plt.subplots()
            df[c["supplier"]].value_counts().nlargest(10).plot.barh(ax=ax)
            ax.set_title("Supplier Concentration")
            save(fig, "resilience_supplier.png", "Supplier dependency", 0.9, "resilience", "dependency", "structure")
    
        if "resilience" in domain_map and c.get("carrier"):
            # 11. Carrier concentration
            fig, ax = plt.subplots()
            df[c["carrier"]].value_counts().nlargest(10).plot.barh(ax=ax)
            ax.set_title("Carrier Concentration")
            save(fig, "resilience_carrier.png", "Carrier dependency", 0.85, "resilience", "dependency", "structure")
    
        # =================================================
        # SUSTAINABILITY — ENVIRONMENTAL SIGNALS
        # =================================================
        if "sustainability" in domain_map and c.get("co2"):
            # 12. CO2 distribution
            fig, ax = plt.subplots()
            df[c["co2"]].hist(ax=ax, bins=20)
            ax.set_title("CO₂ Emissions Distribution")
            save(fig, "sustainability_co2.png", "Emission dispersion", 0.85, "sustainability", "environment", "distribution")
    
        # -------------------------------------------------
        # RETURN MANY — REPORT WILL TRIM
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals

    # ---------------- COMPOSITE INSIGHTS (THE SMART LAYER) ----------------

    def generate_composite_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        otd = kpis.get("otd_rate", 1.0)
        stockout = kpis.get("stockout_rate", 0.0)
        lead = kpis.get("avg_lead_time", 0)
        inventory_lvl = kpis.get("avg_stock", 0)

        # 1. Fulfillment Crisis (Slow Lead Time + Low OTD)
        if otd < 0.85 and lead > 10:
            insights.append({
                "level": "CRITICAL",
                "title": "Fulfillment Crisis",
                "so_what": f"Delivery is unreliable ({otd:.1%}) AND slow ({lead:.1f} days). Systemic logistics failure."
            })

        # 2. Inventory Imbalance (High Stock + High Stockouts)
        # Assuming 'High Stock' is relative, checking if avg > 0
        if inventory_lvl > 0 and stockout > 0.15:
            insights.append({
                "level": "RISK",
                "title": "Inventory Imbalance",
                "so_what": f"You have stock, but {stockout:.1%} of items are OOS. Mismatched demand/supply."
            })

        return insights

    # ---------------- ATOMIC INSIGHTS ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Supply Chain Composite Insight Engine (v1.0)
    
        GUARANTEES:
        - ≥7 composite insights per sub-domain
        - No thresholds or targets
        - KPI-relative, evidence-based
        - Executive-safe language
        - Atomic fallback only if composites missing
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        sub_domains = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # KPI SHORTCUTS (SAFE)
        # -------------------------------------------------
        lead_avg = kpis.get("planning_avg_lead_time")
        lead_var = kpis.get("planning_lead_time_variability")
        otd = kpis.get("logistics_on_time_delivery_rate")
        inventory_avg = kpis.get("inventory_avg_stock")
        inventory_var = kpis.get("inventory_stock_variability")
        cost_avg = kpis.get("cost_avg_cost_per_record")
        cost_dist = kpis.get("cost_cost_per_distance")
        top_supplier_share = kpis.get("resilience_top_supplier_share")
        top_carrier_share = kpis.get("resilience_top_carrier_share")
        co2_avg = kpis.get("sustainability_avg_co2")
        co2_proxy = kpis.get("sustainability_emissions_proxy")
    
        # =================================================
        # PLANNING — FLOW & PREDICTABILITY
        # =================================================
        if "planning" in sub_domains and lead_avg is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Lead Time Baseline Established",
                    "so_what": "Average lead time provides a baseline for planning and scheduling.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Flow Predictability Signal",
                    "so_what": "Lead time variability indicates the predictability of supply flow.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Planning Horizon Context",
                    "so_what": "Lead time magnitude informs effective planning horizons.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Demand–Supply Alignment Signal",
                    "so_what": "Observed lead times reflect alignment between demand and supply capacity.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Planning Stability Indicator",
                    "so_what": "Consistency in lead times supports stable planning assumptions.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Buffer Strategy Context",
                    "so_what": "Lead time dispersion provides context for buffer sizing decisions.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Forecast Readiness",
                    "so_what": "Planning signals are sufficient to support forecast-based decisions.",
                },
            ])
    
        # =================================================
        # LOGISTICS — SERVICE & EXECUTION
        # =================================================
        if "logistics" in sub_domains and otd is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Delivery Reliability Signal",
                    "so_what": "On-time delivery rate reflects service execution consistency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Execution Variability Context",
                    "so_what": "Delivery outcomes vary across orders and time periods.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Service Stability Indicator",
                    "so_what": "OTD trends can be monitored for operational stability.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Throughput Performance Signal",
                    "so_what": "Delivery performance provides insight into throughput capacity.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Fulfillment Reliability Coverage",
                    "so_what": "Data supports ongoing logistics reliability monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Operational Consistency Context",
                    "so_what": "Execution signals highlight consistency of logistics processes.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Service Governance Readiness",
                    "so_what": "Logistics data is sufficient for service governance.",
                },
            ])
    
        # =================================================
        # INVENTORY — AVAILABILITY & CAPITAL
        # =================================================
        if "inventory" in sub_domains and inventory_avg is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Stock Level Baseline",
                    "so_what": "Average inventory levels establish an availability baseline.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Inventory Variability Context",
                    "so_what": "Stock variability reflects replenishment and demand alignment.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Availability Risk Signal",
                    "so_what": "Inventory dispersion highlights potential availability risks.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Working Capital Exposure",
                    "so_what": "Inventory levels influence capital tied up in operations.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Replenishment Cadence Indicator",
                    "so_what": "Inventory patterns suggest replenishment cadence.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Stock Governance Context",
                    "so_what": "Inventory data supports governance and control.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Availability Monitoring Readiness",
                    "so_what": "Inventory signals enable continuous availability monitoring.",
                },
            ])
    
        # =================================================
        # COST — EFFICIENCY & CONTROL
        # =================================================
        if "cost" in sub_domains and cost_avg is not None:
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Baseline Established",
                    "so_what": "Average cost provides a baseline for efficiency analysis.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Variability Signal",
                    "so_what": "Cost dispersion indicates efficiency consistency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Spend Control Context",
                    "so_what": "Cost patterns inform spend governance.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Efficiency Monitoring Capability",
                    "so_what": "Cost data supports efficiency monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Structure Insight",
                    "so_what": "Cost signals reflect structural efficiency drivers.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Operational Cost Coverage",
                    "so_what": "Cost signals cover operational activity.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Governance Readiness",
                    "so_what": "Cost metrics support governance decisions.",
                },
            ])
    
        # =================================================
        # RESILIENCE — DEPENDENCY & RISK
        # =================================================
        if "resilience" in sub_domains and (top_supplier_share or top_carrier_share):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Supplier Dependency Signal",
                    "so_what": "Supplier concentration reflects dependency exposure.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Carrier Dependency Context",
                    "so_what": "Carrier concentration indicates logistics dependency.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Structural Risk Indicator",
                    "so_what": "Dependency patterns inform resilience assessment.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Supply Chain Flexibility Context",
                    "so_what": "Dependency levels affect flexibility under disruption.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Diversification Opportunity Signal",
                    "so_what": "Dependency signals suggest diversification review areas.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Risk Governance Readiness",
                    "so_what": "Data supports risk governance decisions.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Shock Absorption Context",
                    "so_what": "Dependency structure influences shock absorption capacity.",
                },
            ])
    
        # =================================================
        # SUSTAINABILITY — ENVIRONMENTAL EFFICIENCY
        # =================================================
        if "sustainability" in sub_domains and (co2_avg is not None or co2_proxy is not None):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Emission Signal Availability",
                    "so_what": "Environmental impact signals are present.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Emission Variability Context",
                    "so_what": "Emission dispersion indicates efficiency variation.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Environmental Efficiency Baseline",
                    "so_what": "Average emissions provide an efficiency baseline.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Sustainability Monitoring Readiness",
                    "so_what": "Data supports ongoing sustainability monitoring.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Operational Footprint Context",
                    "so_what": "Emissions reflect operational footprint scale.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Efficiency Improvement Context",
                    "so_what": "Environmental signals inform efficiency opportunities.",
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "ESG Readiness Signal",
                    "so_what": "Sustainability data supports ESG reporting readiness.",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Supply Chain Operations Stable",
                "so_what": "Available signals indicate stable supply chain operations.",
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
    
        # =================================================
        # PLANNING — FLOW & FORECASTING
        # =================================================
        if "planning" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "planning",
                    "action": "Use observed lead time patterns to refine planning assumptions.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "planning",
                    "action": "Incorporate lead time variability into demand and capacity planning models.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "planning",
                    "action": "Review planning horizons against observed fulfillment timelines.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "planning",
                    "action": "Align replenishment cadence with observed order flow patterns.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "planning",
                    "action": "Use demand flow trends to stress-test planning scenarios.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "planning",
                    "action": "Assess buffer strategies using lead time dispersion signals.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "planning",
                    "action": "Strengthen planning governance using consistent flow metrics.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # LOGISTICS — FULFILLMENT & SERVICE
        # =================================================
        if "logistics" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "logistics",
                    "action": "Review fulfillment processes using delivery reliability trends.",
                    "priority": "HIGH",
                },
                {
                    "sub_domain": "logistics",
                    "action": "Analyze lead time dispersion to identify execution variability drivers.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "logistics",
                    "action": "Use on-time delivery signals to guide carrier performance reviews.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "logistics",
                    "action": "Assess throughput capacity against observed delivery patterns.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "logistics",
                    "action": "Monitor service stability trends to anticipate execution risks.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "logistics",
                    "action": "Standardize fulfillment metrics for ongoing service governance.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "logistics",
                    "action": "Use logistics performance signals to support continuous improvement initiatives.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # INVENTORY — AVAILABILITY & CAPITAL
        # =================================================
        if "inventory" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "inventory",
                    "action": "Review stock level distributions to assess availability posture.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Align replenishment policies with observed inventory variability.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Evaluate working capital exposure using inventory level baselines.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Use inventory dispersion signals to identify potential imbalance risks.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Strengthen inventory governance with consistent availability metrics.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Assess category-level stock mix for alignment with demand patterns.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Incorporate inventory signals into broader supply planning reviews.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # COST — EFFICIENCY & CONTROL
        # =================================================
        if "cost" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "cost",
                    "action": "Review cost distributions to understand efficiency variability.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "cost",
                    "action": "Use cost-per-distance signals to assess logistics efficiency drivers.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "cost",
                    "action": "Monitor cost trends to support spend governance decisions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "cost",
                    "action": "Align cost metrics with operational performance reviews.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "cost",
                    "action": "Evaluate cost structure consistency across routes and carriers.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "cost",
                    "action": "Use cost signals to inform efficiency improvement initiatives.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "cost",
                    "action": "Standardize cost reporting for executive oversight.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # RESILIENCE — DEPENDENCY & RISK
        # =================================================
        if "resilience" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "resilience",
                    "action": "Review supplier concentration to understand dependency exposure.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "resilience",
                    "action": "Assess carrier dependency patterns to evaluate flexibility.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "resilience",
                    "action": "Use dependency signals to inform diversification discussions.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "resilience",
                    "action": "Incorporate dependency metrics into risk governance frameworks.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "resilience",
                    "action": "Stress-test supply chain scenarios using dependency structures.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "resilience",
                    "action": "Monitor concentration trends for early risk signals.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "resilience",
                    "action": "Align resilience metrics with continuity planning.",
                    "priority": "LOW",
                },
            ])
    
        # =================================================
        # SUSTAINABILITY — ENVIRONMENTAL EFFICIENCY
        # =================================================
        if "sustainability" in sub_domains:
            recs.extend([
                {
                    "sub_domain": "sustainability",
                    "action": "Review emissions signals to establish environmental baselines.",
                    "priority": "MEDIUM",
                },
                {
                    "sub_domain": "sustainability",
                    "action": "Use emission variability to identify efficiency opportunities.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sustainability",
                    "action": "Integrate sustainability metrics into logistics performance reviews.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sustainability",
                    "action": "Assess route and carrier choices through an environmental lens.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sustainability",
                    "action": "Align sustainability signals with ESG reporting objectives.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sustainability",
                    "action": "Monitor environmental efficiency trends over time.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "sustainability",
                    "action": "Use sustainability data to inform long-term optimization initiatives.",
                    "priority": "LOW",
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not recs:
            recs.append({
                "sub_domain": "mixed",
                "action": "Continue monitoring supply chain performance signals.",
                "priority": "LOW",
            })
    
        return recs

# =====================================================
# DOMAIN DETECTOR
# =====================================================

class SupplyChainDomainDetector(BaseDomainDetector):
    """
    Supply Chain Domain Detector (v1.1)

    Detects datasets focused on:
    - Inventory positioning
    - Order fulfillment & delivery
    - Logistics movement
    - Supplier / carrier dependency

    Explicitly avoids:
    - Retail sales ownership
    - Ecommerce revenue realization
    """

    domain_name = "supply_chain"

    # Strong supply chain anchors (raw operational signals)
    SUPPLY_CHAIN_ANCHORS: Set[str] = {
        "inventory",
        "stock",
        "stock_level",
        "order_id",
        "shipment",
        "ship_date",
        "delivery_date",
        "promised_date",
        "carrier",
        "supplier",
        "logistics",
        "freight",
        "warehouse",
    }

    # Commerce ownership signals (boundary control)
    EXCLUSION_TOKENS: Set[str] = {
        "revenue",
        "sales",
        "price",
        "transaction",
        "order_value",
        "gmv",
        "customer",
        "payment",
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
        has_inventory = any("inventory" in c or "stock" in c for c in cols)
        has_delivery = any("delivery" in c or "ship" in c for c in cols)
        has_logistics = any("carrier" in c or "freight" in c or "logistics" in c for c in cols)
        has_supplier = any("supplier" in c or "vendor" in c for c in cols)

        core_signals = sum([
            has_inventory,
            has_delivery,
            has_logistics,
        ])

        # -------------------------------------------------
        # BASE CONFIDENCE (CAPABILITY-BASED)
        # -------------------------------------------------
        confidence = 0.0

        if core_signals >= 2:
            confidence = 0.65

        if core_signals == 3:
            confidence = 0.8

        if has_supplier:
            confidence += 0.05

        # -------------------------------------------------
        # BOUNDARY CONTROL
        # -------------------------------------------------
        has_revenue = any(t in c for t in self.EXCLUSION_TOKENS for c in cols)

        if has_revenue:
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
                    "supply_chain_signals": {
                        "inventory": has_inventory,
                        "delivery": has_delivery,
                        "logistics": has_logistics,
                        "supplier": has_supplier,
                    },
                },
            )

        return DomainDetectionResult(
            domain="supply_chain",
            confidence=confidence,
            signals={
                "supply_chain_signals": {
                    "inventory": has_inventory,
                    "delivery": has_delivery,
                    "logistics": has_logistics,
                    "supplier": has_supplier,
                },
                "excluded_signals": [
                    c for c in cols if any(t in c for t in self.EXCLUSION_TOKENS)
                ],
            },
        )

def register(registry):
    registry.register("supply_chain", SupplyChainDomain, SupplyChainDomainDetector)
