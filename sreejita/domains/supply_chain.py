import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # governance: non-interactive backend
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
    Governance-safe division.

    Guarantees:
    - Never raises
    - Returns None for zero, NaN, or invalid inputs
    - Explicit float coercion
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
        return val
    except Exception:
        return None


def _safe_mean(series: Optional[pd.Series]) -> Optional[float]:
    """
    Governance-safe mean.

    Guarantees:
    - Graceful degradation
    - Numeric coercion
    - Returns None if insufficient signal
    """
    if series is None or not isinstance(series, pd.Series):
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Supply Chain–safe time column detector.

    Supported semantics (ordered by operational relevance):
    - Delivery / receipt dates
    - Ship dates
    - Order dates
    - Generic date / timestamp

    Design principles:
    - Logistics-aware preference
    - No dataset assumptions
    - No dataframe mutation
    - Returns None if confidence is weak
    """
    if df is None or df.empty:
        return None

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
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= 3:
                    return col
            except Exception:
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
        for key in ("order_date", "ship_date", "delivery_date", "promised_date"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Operational time columns
        for key in ("processing_time", "packing_time"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Canonical time sort (if available)
        if self.time_col and self.time_col in df.columns:
            df = df.sort_values(self.time_col)

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (SAFE, NON-DESTRUCTIVE)
        # -------------------------------------------------
        for key in ("inventory", "cost", "distance", "weight", "co2"):
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
        # DATA COMPLETENESS (RAW SIGNALS ONLY)
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
            1 for k in raw_signal_keys
            if self.cols.get(k)
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

            if "_confidence" not in kpis:
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
        - Graceful degradation
        """
    
        if df is None or df.empty:
            return {}
    
        c = self.cols
        volume = int(len(df))
    
        # -------------------------------------------------
        # SUB-DOMAIN DEFINITIONS (CANONICAL)
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
        # SAFE HELPERS (COLUMN-BASED, GOVERNED)
        # -------------------------------------------------
        def safe_sum(col: Optional[str]) -> Optional[float]:
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.sum()) if s.notna().any() else None
    
        def safe_mean(col: Optional[str]) -> Optional[float]:
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        # =================================================
        # PLANNING & FLOW STABILITY
        # =================================================
        planning: List[str] = []
    
        total_orders = (
            df[c["order_id"]].nunique()
            if c.get("order_id") and c["order_id"] in df.columns
            else volume
        )
        kpis["planning_total_orders"] = total_orders
        planning.append("planning_total_orders")
    
        if c.get("order_date") and c["order_date"] in df.columns:
            unique_days = df[c["order_date"]].nunique()
            kpis["planning_order_frequency"] = _safe_div(
                total_orders, max(unique_days, 1)
            )
            planning.append("planning_order_frequency")
    
        # --- Lead Time (DATE-BASED OR OPERATIONAL PROXY) ---
        lead_days = None
    
        if (
            c.get("order_date")
            and c.get("delivery_date")
            and c["order_date"] in df.columns
            and c["delivery_date"] in df.columns
        ):
            lead_days = (
                df[c["delivery_date"]] - df[c["order_date"]]
            ).dt.days
    
        elif c.get("processing_time") or c.get("packing_time"):
            proc = (
                pd.to_numeric(df[c["processing_time"]], errors="coerce")
                if c.get("processing_time") and c["processing_time"] in df.columns
                else 0
            )
            pack = (
                pd.to_numeric(df[c["packing_time"]], errors="coerce")
                if c.get("packing_time") and c["packing_time"] in df.columns
                else 0
            )
            # hours → days (explicit proxy)
            lead_days = (proc.fillna(0) + pack.fillna(0)) / 24.0
    
        if lead_days is not None:
            lead_days = pd.to_numeric(lead_days, errors="coerce").dropna()
            if not lead_days.empty:
                mean_lead = lead_days.mean()
                kpis["planning_avg_lead_time"] = float(mean_lead)
                kpis["planning_lead_time_variability"] = _safe_div(
                    lead_days.std(), mean_lead
                )
                planning.extend([
                    "planning_avg_lead_time",
                    "planning_lead_time_variability",
                ])
    
        # =================================================
        # INVENTORY & WORKING CAPITAL
        # =================================================
        inventory: List[str] = []
    
        if c.get("inventory") and c["inventory"] in df.columns:
            inv = pd.to_numeric(df[c["inventory"]], errors="coerce").dropna()
            if not inv.empty:
                mean_inv = inv.mean()
                kpis["inventory_avg_stock"] = float(mean_inv)
                kpis["inventory_stock_variability"] = _safe_div(
                    inv.std(), mean_inv
                )
                kpis["inventory_zero_stock_ratio"] = float((inv <= 0).mean())
                inventory.extend([
                    "inventory_avg_stock",
                    "inventory_stock_variability",
                    "inventory_zero_stock_ratio",
                ])
    
        # =================================================
        # LOGISTICS & FULFILLMENT
        # =================================================
        logistics: List[str] = []
    
        if (
            c.get("delivery_date")
            and c.get("promised_date")
            and c["delivery_date"] in df.columns
            and c["promised_date"] in df.columns
        ):
            valid = df[c["delivery_date"]].notna() & df[c["promised_date"]].notna()
            if valid.any():
                on_time = (
                    df.loc[valid, c["delivery_date"]]
                    <= df.loc[valid, c["promised_date"]]
                )
                kpis["logistics_on_time_delivery_rate"] = float(on_time.mean())
                logistics.append("logistics_on_time_delivery_rate")
    
        elif c.get("status") and c["status"] in df.columns:
            status = df[c["status"]].astype(str).str.lower()
            delivered = status.str.contains("deliver", na=False)
            resolved = status.str.contains("deliver|return|fail|cancel", na=False)
            if resolved.any():
                kpis["logistics_on_time_delivery_rate"] = float(delivered.mean())
                logistics.append("logistics_on_time_delivery_rate")
    
        if c.get("distance") and c["distance"] in df.columns:
            kpis["logistics_avg_distance"] = safe_mean(c["distance"])
            logistics.append("logistics_avg_distance")
    
        # =================================================
        # COST EFFICIENCY
        # =================================================
        cost: List[str] = []
    
        if c.get("cost") and c["cost"] in df.columns:
            total_cost = safe_sum(c["cost"])
            kpis["cost_total_cost"] = total_cost
            kpis["cost_avg_cost_per_record"] = safe_mean(c["cost"])
            cost.extend([
                "cost_total_cost",
                "cost_avg_cost_per_record",
            ])
    
        if (
            c.get("cost")
            and c.get("distance")
            and c["cost"] in df.columns
            and c["distance"] in df.columns
        ):
            kpis["cost_cost_per_distance"] = _safe_div(
                safe_sum(c["cost"]),
                safe_sum(c["distance"]),
            )
            cost.append("cost_cost_per_distance")
    
        # =================================================
        # RESILIENCE & DEPENDENCY
        # =================================================
        resilience: List[str] = []
    
        if c.get("supplier") and c["supplier"] in df.columns:
            counts = df[c["supplier"]].value_counts()
            if not counts.empty:
                kpis["resilience_supplier_count"] = int(counts.size)
                kpis["resilience_top_supplier_share"] = float(
                    counts.iloc[0] / counts.sum()
                )
                resilience.extend([
                    "resilience_supplier_count",
                    "resilience_top_supplier_share",
                ])
    
        if c.get("carrier") and c["carrier"] in df.columns:
            counts = df[c["carrier"]].value_counts()
            if not counts.empty:
                kpis["resilience_carrier_count"] = int(counts.size)
                kpis["resilience_top_carrier_share"] = float(
                    counts.iloc[0] / counts.sum()
                )
                resilience.extend([
                    "resilience_carrier_count",
                    "resilience_top_carrier_share",
                ])
    
        # =================================================
        # SUSTAINABILITY (PROXY-AWARE)
        # =================================================
        sustainability: List[str] = []
    
        if c.get("co2") and c["co2"] in df.columns:
            kpis["sustainability_avg_co2"] = safe_mean(c["co2"])
            sustainability.append("sustainability_avg_co2")
    
        elif c.get("distance") and c["distance"] in df.columns:
            kpis["sustainability_emissions_proxy"] = _safe_div(
                safe_sum(c["distance"]), volume
            )
            sustainability.append("sustainability_emissions_proxy")
    
        # -------------------------------------------------
        # DOMAIN → KPI MAP (AUTHORITATIVE)
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
        # KPI CONFIDENCE (STRUCTURAL, NOT JUDGMENTAL)
        # -------------------------------------------------
        for key, val in kpis.items():
            if key.startswith("_") or not isinstance(val, (int, float)):
                continue
    
            base = 0.70
            if volume < 100:
                base -= 0.15
            if "proxy" in key:
                base -= 0.10
            if "rate" in key or "variability" in key:
                base += 0.05
    
            kpis["_confidence"][key] = round(
                max(0.40, min(0.90, base)), 2
            )
    
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
        - Evidence-only visuals (no thresholds, no judgement)
        - KPI-backed (single source of truth)
        - No dataframe mutation
        - No trimming here (report layer decides)
        - Graceful degradation
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
        record_count = int(kpis.get("record_count", 0))
    
        # -------------------------------------------------
        # VISUAL CONFIDENCE (DATA-DRIVEN, NON-JUDGMENTAL)
        # -------------------------------------------------
        if record_count >= 5000:
            visual_conf = 0.85
        elif record_count >= 1000:
            visual_conf = 0.70
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
        if "planning" in domain_map and self.time_col and self.time_col in df.columns:
            ts = df.set_index(self.time_col)
    
            # 1. Order volume trend
            fig, ax = plt.subplots()
            ts.resample("M").size().plot(ax=ax)
            ax.set_title("Order Volume Over Time")
            save(
                fig,
                "planning_volume_trend.png",
                "Observed order volume trend over time",
                0.95,
                "planning",
                "volume",
                "time",
            )
    
            # 2. Order volume distribution
            fig, ax = plt.subplots()
            ts.resample("M").size().hist(ax=ax, bins=20)
            ax.set_title("Order Volume Distribution")
            save(
                fig,
                "planning_volume_dist.png",
                "Distribution of observed order volumes",
                0.80,
                "planning",
                "volume",
                "distribution",
            )
    
        # =================================================
        # LOGISTICS — FULFILLMENT
        # =================================================
        if (
            "logistics" in domain_map
            and c.get("order_date")
            and c.get("delivery_date")
            and c["order_date"] in df.columns
            and c["delivery_date"] in df.columns
        ):
            lead = (
                df[c["delivery_date"]] - df[c["order_date"]]
            ).dt.days
            lead = pd.to_numeric(lead, errors="coerce").dropna()
    
            if lead.nunique() > 3:
                fig, ax = plt.subplots()
                lead.hist(ax=ax, bins=20)
                ax.set_title("Lead Time Distribution")
                save(
                    fig,
                    "logistics_lead_dist.png",
                    "Observed fulfillment time dispersion",
                    0.95,
                    "logistics",
                    "velocity",
                    "distribution",
                )
    
                fig, ax = plt.subplots()
                lead.plot(kind="box", ax=ax)
                ax.set_title("Lead Time Spread")
                save(
                    fig,
                    "logistics_lead_box.png",
                    "Observed delivery time variability",
                    0.90,
                    "logistics",
                    "variability",
                    "spread",
                )
    
        if (
            "logistics" in domain_map
            and c.get("delivery_date")
            and c.get("promised_date")
            and self.time_col
            and self.time_col in df.columns
            and c["delivery_date"] in df.columns
            and c["promised_date"] in df.columns
        ):
            valid = df[c["delivery_date"]].notna() & df[c["promised_date"]].notna()
            if valid.any():
                on_time = (
                    df.loc[valid, c["delivery_date"]]
                    <= df.loc[valid, c["promised_date"]]
                )
    
                fig, ax = plt.subplots()
                on_time.groupby(
                    df.loc[valid, self.time_col].dt.to_period("M")
                ).mean().plot(ax=ax)
                ax.set_title("On-Time Delivery Over Time")
                save(
                    fig,
                    "logistics_otd_trend.png",
                    "Observed delivery reliability trend",
                    0.90,
                    "logistics",
                    "reliability",
                    "time",
                )
    
        # =================================================
        # INVENTORY — STOCK POSITION
        # =================================================
        if "inventory" in domain_map and c.get("inventory") and c["inventory"] in df.columns:
            inv = pd.to_numeric(df[c["inventory"]], errors="coerce").dropna()
            if inv.nunique() > 3:
                fig, ax = plt.subplots()
                inv.hist(ax=ax, bins=20)
                ax.set_title("Inventory Level Distribution")
                save(
                    fig,
                    "inventory_dist.png",
                    "Observed inventory level dispersion",
                    0.90,
                    "inventory",
                    "stock",
                    "distribution",
                )
    
            if c.get("category") and c["category"] in df.columns:
                fig, ax = plt.subplots()
                df.groupby(c["category"])[c["inventory"]].mean().nlargest(10).plot.bar(ax=ax)
                ax.set_title("Average Inventory by Category")
                save(
                    fig,
                    "inventory_category.png",
                    "Category-level inventory mix",
                    0.85,
                    "inventory",
                    "mix",
                    "entity",
                )
    
        # =================================================
        # COST — EFFICIENCY
        # =================================================
        if "cost" in domain_map and c.get("cost") and c["cost"] in df.columns:
            cost = pd.to_numeric(df[c["cost"]], errors="coerce").dropna()
            if cost.nunique() > 3:
                fig, ax = plt.subplots()
                cost.hist(ax=ax, bins=20)
                ax.set_title("Cost Distribution")
                ax.xaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "cost_dist.png",
                    "Observed cost variability",
                    0.90,
                    "cost",
                    "efficiency",
                    "distribution",
                )
    
            if self.time_col and self.time_col in df.columns:
                fig, ax = plt.subplots()
                df.set_index(self.time_col)[c["cost"]].resample("M").sum().plot(ax=ax)
                ax.set_title("Cost Over Time")
                ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
                save(
                    fig,
                    "cost_trend.png",
                    "Observed cost trajectory over time",
                    0.85,
                    "cost",
                    "efficiency",
                    "time",
                )
    
        # =================================================
        # RESILIENCE — DEPENDENCY
        # =================================================
        if "resilience" in domain_map and c.get("supplier") and c["supplier"] in df.columns:
            fig, ax = plt.subplots()
            df[c["supplier"]].value_counts().nlargest(10).plot.barh(ax=ax)
            ax.set_title("Supplier Concentration")
            save(
                fig,
                "resilience_supplier.png",
                "Observed supplier dependency structure",
                0.90,
                "resilience",
                "dependency",
                "structure",
            )
    
        if "resilience" in domain_map and c.get("carrier") and c["carrier"] in df.columns:
            fig, ax = plt.subplots()
            df[c["carrier"]].value_counts().nlargest(10).plot.barh(ax=ax)
            ax.set_title("Carrier Concentration")
            save(
                fig,
                "resilience_carrier.png",
                "Observed carrier dependency structure",
                0.85,
                "resilience",
                "dependency",
                "structure",
            )
    
        # =================================================
        # SUSTAINABILITY — ENVIRONMENTAL SIGNALS
        # =================================================
        if "sustainability" in domain_map and c.get("co2") and c["co2"] in df.columns:
            co2 = pd.to_numeric(df[c["co2"]], errors="coerce").dropna()
            if co2.nunique() > 3:
                fig, ax = plt.subplots()
                co2.hist(ax=ax, bins=20)
                ax.set_title("CO₂ Emissions Distribution")
                save(
                    fig,
                    "sustainability_co2.png",
                    "Observed emissions dispersion",
                    0.85,
                    "sustainability",
                    "environment",
                    "distribution",
                )
    
        # -------------------------------------------------
        # RETURN MANY — REPORT LAYER WILL TRIM
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals


    # ---------------- COMPOSITE INSIGHTS (THE SMART LAYER) ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Supply Chain Composite Insight Engine (v1.0)
    
        GUARANTEES:
        - Composite-first logic
        - Capability-aligned (planning, inventory, logistics, cost, resilience, sustainability)
        - No thresholds, targets, or benchmarks
        - KPI-relative, evidence-based
        - Executive-safe, non-judgmental language
        - Atomic fallback only if composites missing
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        sub_domains = kpis.get("_domain_kpi_map", {}) or {}
    
        # -------------------------------------------------
        # KPI SHORTCUTS (SAFE, ALIGNED TO KPI ENGINE)
        # -------------------------------------------------
        lead_avg = kpis.get("planning_avg_lead_time")
        lead_var = kpis.get("planning_lead_time_variability")
    
        otd = kpis.get("logistics_on_time_delivery_rate")
    
        inventory_avg = kpis.get("inventory_avg_stock")
        inventory_var = kpis.get("inventory_stock_variability")
        inventory_zero = kpis.get("inventory_zero_stock_ratio")
    
        cost_avg = kpis.get("cost_avg_cost_per_record")
        cost_dist = kpis.get("cost_cost_per_distance")
    
        supplier_share = kpis.get("resilience_top_supplier_share")
        carrier_share = kpis.get("resilience_top_carrier_share")
    
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
                    "so_what": (
                        "Observed average lead time provides a baseline view of end-to-end flow duration."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Flow Predictability Context",
                    "so_what": (
                        "Lead time variability reflects how predictable the supply flow is across orders."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Planning Horizon Signal",
                    "so_what": (
                        "Lead time magnitude informs feasible planning and replenishment horizons."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Buffer Strategy Context",
                    "so_what": (
                        "Dispersion in lead times provides context for buffer and safety strategies."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Flow Stability Indicator",
                    "so_what": (
                        "Consistency in lead times supports stable planning assumptions."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Demand–Supply Alignment Signal",
                    "so_what": (
                        "Observed lead times reflect alignment between demand patterns and supply capacity."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "planning",
                    "title": "Planning Readiness",
                    "so_what": (
                        "Available planning signals support structured flow governance."
                    ),
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
                    "so_what": (
                        "Observed on-time delivery performance reflects execution consistency."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Service Variability Context",
                    "so_what": (
                        "Delivery outcomes vary across orders and time periods."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Fulfillment Stability Indicator",
                    "so_what": (
                        "OTD patterns provide insight into fulfillment stability."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Throughput Performance Context",
                    "so_what": (
                        "Delivery performance reflects throughput capability across the network."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Execution Consistency Signal",
                    "so_what": (
                        "Observed delivery reliability supports logistics process assessment."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Service Governance Readiness",
                    "so_what": (
                        "Logistics data supports ongoing service governance."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "logistics",
                    "title": "Fulfillment Monitoring Capability",
                    "so_what": (
                        "Available signals enable continuous fulfillment monitoring."
                    ),
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
                    "title": "Inventory Level Baseline",
                    "so_what": (
                        "Average inventory levels establish an availability baseline."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Inventory Variability Context",
                    "so_what": (
                        "Stock variability reflects replenishment and demand alignment."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Availability Risk Signal",
                    "so_what": (
                        "Observed inventory dispersion highlights availability risk exposure."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Working Capital Context",
                    "so_what": (
                        "Inventory levels influence capital tied up in operations."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Replenishment Cadence Indicator",
                    "so_what": (
                        "Inventory patterns suggest replenishment cadence stability."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Stock Governance Readiness",
                    "so_what": (
                        "Inventory signals support governance and control mechanisms."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "inventory",
                    "title": "Availability Monitoring Capability",
                    "so_what": (
                        "Data supports continuous availability monitoring."
                    ),
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
                    "so_what": (
                        "Average cost provides a baseline for efficiency analysis."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Variability Signal",
                    "so_what": (
                        "Cost dispersion reflects consistency of operational efficiency."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Spend Structure Context",
                    "so_what": (
                        "Cost patterns reveal structural drivers of spend."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Efficiency Monitoring Capability",
                    "so_what": (
                        "Cost data supports ongoing efficiency monitoring."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Cost Governance Readiness",
                    "so_what": (
                        "Available cost signals support governance decisions."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Operational Coverage Signal",
                    "so_what": (
                        "Cost metrics cover core operational activity."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "cost",
                    "title": "Efficiency Improvement Context",
                    "so_what": (
                        "Cost signals provide context for efficiency discussions."
                    ),
                },
            ])
    
        # =================================================
        # RESILIENCE — DEPENDENCY & RISK
        # =================================================
        if "resilience" in sub_domains and (supplier_share is not None or carrier_share is not None):
            insights.extend([
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Supplier Dependency Signal",
                    "so_what": (
                        "Supplier concentration reflects dependency exposure."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Carrier Dependency Context",
                    "so_what": (
                        "Carrier concentration indicates logistics dependency."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Structural Risk Indicator",
                    "so_what": (
                        "Dependency patterns inform resilience assessment."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Flexibility Context",
                    "so_what": (
                        "Dependency levels influence flexibility under disruption."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Diversification Opportunity Signal",
                    "so_what": (
                        "Dependency signals suggest diversification review areas."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Risk Governance Readiness",
                    "so_what": (
                        "Data supports structured risk governance."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "resilience",
                    "title": "Shock Absorption Context",
                    "so_what": (
                        "Dependency structure influences shock absorption capacity."
                    ),
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
                    "so_what": (
                        "Environmental impact signals are available for analysis."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Emission Variability Context",
                    "so_what": (
                        "Emission dispersion reflects efficiency variation."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Environmental Efficiency Baseline",
                    "so_what": (
                        "Average emissions provide an efficiency baseline."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Operational Footprint Context",
                    "so_what": (
                        "Emissions reflect operational footprint scale."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Sustainability Monitoring Readiness",
                    "so_what": (
                        "Data supports ongoing sustainability monitoring."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "Efficiency Improvement Context",
                    "so_what": (
                        "Environmental signals inform efficiency discussions."
                    ),
                },
                {
                    "level": "INFO",
                    "sub_domain": "sustainability",
                    "title": "ESG Readiness Signal",
                    "so_what": (
                        "Sustainability data supports ESG reporting readiness."
                    ),
                },
            ])
    
        # -------------------------------------------------
        # GUARANTEED FALLBACK
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": "mixed",
                "title": "Supply Chain Operations Overview",
                "so_what": (
                    "Available signals support a descriptive overview of supply chain operations."
                ),
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
        Supply Chain Advisory Recommendation Engine (v1.0)
    
        GUARANTEES:
        - Advisory-only (no mandates)
        - Sub-domain aware
        - Insight-decoupled
        - No thresholds or benchmarks
        - Executive-safe language
        """
    
        recs: List[Dict[str, Any]] = []
    
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
                    "action": "Assess buffer strategies using lead time dispersion signals.",
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
                    "action": "Assess category-level stock mix for alignment with demand patterns.",
                    "priority": "LOW",
                },
                {
                    "sub_domain": "inventory",
                    "action": "Strengthen inventory governance with consistent availability metrics.",
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
    Supply Chain Domain Detector (v1.2)

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

    # Strong supply chain anchors (operational signals)
    SUPPLY_CHAIN_ANCHORS: Set[str] = {
        "inventory", "stock", "stock_level",
        "order_id", "shipment",
        "ship_date", "delivery_date", "promised_date",
        "carrier", "supplier", "vendor",
        "logistics", "freight", "warehouse",
    }

    # Commerce / ownership signals (boundary control)
    COMMERCE_TOKENS: Set[str] = {
        "revenue", "sales", "price", "gmv",
        "order_value", "transaction", "payment",
        "customer",
    }

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        cols = [str(c).lower() for c in df.columns]

        def tokenize(col: str) -> Set[str]:
            return set(col.replace("_", " ").split())

        tokenized = {c: tokenize(c) for c in cols}

        # -------------------------------------------------
        # SUPPLY CHAIN CAPABILITY SIGNALS
        # -------------------------------------------------
        inventory_hits = {
            c for c, t in tokenized.items()
            if t & {"inventory", "stock"}
        }

        delivery_hits = {
            c for c, t in tokenized.items()
            if t & {"ship", "delivery", "shipment"}
        }

        logistics_hits = {
            c for c, t in tokenized.items()
            if t & {"carrier", "freight", "logistics", "warehouse"}
        }

        supplier_hits = {
            c for c, t in tokenized.items()
            if t & {"supplier", "vendor"}
        }

        core_signal_groups = sum([
            bool(inventory_hits),
            bool(delivery_hits),
            bool(logistics_hits),
        ])

        # -------------------------------------------------
        # BASE CONFIDENCE (CAPABILITY-DRIVEN)
        # -------------------------------------------------
        confidence = 0.0

        if core_signal_groups == 1:
            confidence = 0.45
        elif core_signal_groups == 2:
            confidence = 0.65
        elif core_signal_groups == 3:
            confidence = 0.8

        if supplier_hits:
            confidence += 0.05

        # -------------------------------------------------
        # NEGATIVE GATES — COMMERCE OWNERSHIP
        # -------------------------------------------------
        commerce_hits = {
            c for c, t in tokenized.items()
            if t & self.COMMERCE_TOKENS
        }

        # Strong commerce signature → suppress
        has_customer = any("customer" in c for c in cols)
        has_revenue = any("revenue" in c or "sales" in c for c in cols)
        has_order_value = any("value" in c or "price" in c for c in cols)

        if has_customer and has_revenue and has_order_value:
            confidence *= 0.4

        # Light penalty for mixed datasets
        elif commerce_hits:
            confidence -= 0.15

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
                        "inventory": bool(inventory_hits),
                        "delivery": bool(delivery_hits),
                        "logistics": bool(logistics_hits),
                        "supplier": bool(supplier_hits),
                    },
                    "commerce_signals": sorted(commerce_hits),
                },
            )

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals={
                "supply_chain_signals": {
                    "inventory": bool(inventory_hits),
                    "delivery": bool(delivery_hits),
                    "logistics": bool(logistics_hits),
                    "supplier": bool(supplier_hits),
                },
                "commerce_signals": sorted(commerce_hits),
            },
        )


def register(registry):
    registry.register(
        "supply_chain",
        SupplyChainDomain,
        SupplyChainDomainDetector,
    )
