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
# HELPERS
# =====================================================

def _safe_div(n, d):
    if d in (0, None) or pd.isna(d):
        return None
    return n / d

def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Manufacturing time detector (Production Date, Shift Start, etc.)."""
    candidates = ["date", "timestamp", "production_date", "shift_start", "run_time"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except:
                continue
    return None


# =====================================================
# MANUFACTURING DOMAIN (UNIVERSAL 10/10)
# =====================================================

class ManufacturingDomain(BaseDomain):
    name = "manufacturing"
    description = "Universal Manufacturing Intelligence (OEE, Quality, Downtime, Production)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        
        # 1. Resolve columns ONCE.
        self.cols = {
            # Production
            "order_id": resolve_column(df, "order_id") or resolve_column(df, "production_order"),
            "product": resolve_column(df, "product") or resolve_column(df, "item") or resolve_column(df, "material"),
            "machine": resolve_column(df, "machine") or resolve_column(df, "equipment") or resolve_column(df, "line"),
            "quantity": resolve_column(df, "quantity") or resolve_column(df, "produced_qty") or resolve_column(df, "output"),
            "target": resolve_column(df, "target_qty") or resolve_column(df, "planned_qty"),
            
            # Quality
            "defects": resolve_column(df, "defects") or resolve_column(df, "rejected_qty") or resolve_column(df, "scrap"),
            "quality_score": resolve_column(df, "quality_score"),
            
            # Time / Efficiency
            "cycle_time": resolve_column(df, "cycle_time") or resolve_column(df, "run_time"),
            "downtime": resolve_column(df, "downtime") or resolve_column(df, "stop_time") or resolve_column(df, "breakdown"),
            "uptime": resolve_column(df, "uptime") or resolve_column(df, "operating_time"),
            "shift": resolve_column(df, "shift"),
            
            # Cost
            "cost": resolve_column(df, "cost") or resolve_column(df, "production_cost")
        }

        # 2. Numeric Safety
        for m in ["quantity", "target", "defects", "cycle_time", "downtime", "uptime", "cost"]:
            if self.cols[m]:
                df[self.cols[m]] = pd.to_numeric(df[self.cols[m]], errors='coerce').fillna(0)

        # 3. Date Cleaning
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

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
        Execute the manufacturing pipeline with stable, safe outputs.

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

        # 1. Production Volume
        if c["quantity"]:
            kpis["total_production"] = df[c["quantity"]].sum()
            
            # Yield Rate (Good Parts / Total Parts)
            if c["defects"]:
                total = df[c["quantity"]].sum()
                defects = df[c["defects"]].sum()
                kpis["defect_rate"] = _safe_div(defects, total)
                kpis["yield_rate"] = 1.0 - (kpis.get("defect_rate") or 0)

        # 2. OEE (Overall Equipment Effectiveness) Proxy
        # Availability * Performance * Quality
        # Ideally needs specific columns, but we can approximate components
        if c["uptime"] and c["downtime"]:
            total_time = df[c["uptime"]].sum() + df[c["downtime"]].sum()
            kpis["availability"] = _safe_div(df[c["uptime"]].sum(), total_time)

        if c["quantity"] and c["target"]:
            kpis["performance"] = _safe_div(df[c["quantity"]].sum(), df[c["target"]].sum())

        if kpis.get("availability") and kpis.get("performance") and kpis.get("yield_rate"):
            kpis["oee"] = kpis["availability"] * kpis["performance"] * kpis["yield_rate"]

        # 3. Efficiency
        if c["cycle_time"] and c["quantity"]:
            # Avg time per unit
            kpis["avg_cycle_time"] = _safe_div(df[c["cycle_time"]].sum(), df[c["quantity"]].sum())

        # 4. Downtime Impact
        if c["downtime"]:
            kpis["total_downtime_hours"] = df[c["downtime"]].sum()
            
        # 5. Cost
        if c["cost"] and kpis.get("total_production"):
            kpis["cost_per_unit"] = _safe_div(df[c["cost"]].sum(), kpis["total_production"])

        return kpis

    # ---------------- VISUALS (8 CANDIDATES) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        kpis = self.calculate_kpis(df)

        def save(fig, name, caption, imp, cat):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": imp,
                "category": cat
            })

        def human_fmt(x, _):
            if x >= 1e6: return f"{x/1e6:.1f}M"
            if x >= 1e3: return f"{x/1e3:.0f}K"
            return str(int(x))

        # 1. Production Trend
        if self.time_col and c["quantity"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.set_index(pd.to_datetime(df[self.time_col])).resample('D')[c["quantity"]].sum().plot(ax=ax, color="#1f77b4")
            ax.set_title("Daily Production Output")
            ax.yaxis.set_major_formatter(FuncFormatter(human_fmt))
            save(fig, "prod_trend.png", "Output volume", 0.9, "production")

        # 2. Defect Rate by Machine/Line
        if c["machine"] and c["defects"] and c["quantity"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            # Calculate defect rate per machine
            mach_stats = df.groupby(c["machine"]).agg({c["defects"]: "sum", c["quantity"]: "sum"})
            mach_stats["rate"] = mach_stats[c["defects"]] / mach_stats[c["quantity"]].replace(0, 1)
            
            mach_stats["rate"].nlargest(5).plot(kind="bar", ax=ax, color="red")
            ax.set_title("Defect Rate by Machine (Top 5)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
            # High priority if defects are high
            imp = 0.95 if kpis.get("defect_rate", 0) > 0.05 else 0.8
            save(fig, "defect_machine.png", "Quality hotspots", imp, "quality")

        # 3. Downtime Reasons (Pareto)
        # Assuming 'reason' column exists or using 'machine' as proxy
        downtime_cause = resolve_column(df, "downtime_reason") or c["machine"]
        if c["downtime"] and downtime_cause:
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby(downtime_cause)[c["downtime"]].sum().nlargest(7).plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Top Contributors to Downtime")
            save(fig, "downtime_pareto.png", "Loss analysis", 0.85, "maintenance")

        # 4. Target vs Actual
        if c["target"] and c["quantity"]:
            fig, ax = plt.subplots(figsize=(5, 4))
            totals = [df[c["target"]].sum(), df[c["quantity"]].sum()]
            ax.bar(["Planned", "Actual"], totals, color=["gray", "green"])
            ax.set_title("Production Plan Adherence")
            save(fig, "plan_adherence.png", "Schedule attainment", 0.88, "planning")

        # 5. OEE Components
        if kpis.get("oee"):
            fig, ax = plt.subplots(figsize=(6, 4))
            vals = [kpis["availability"], kpis["performance"], kpis["yield_rate"]]
            ax.bar(["Availability", "Performance", "Quality"], vals, color=["#17becf", "#bcbd22", "#2ca02c"])
            ax.set_ylim(0, 1.1)
            ax.set_title(f"OEE Components (Score: {kpis['oee']:.0%})")
            save(fig, "oee_chart.png", "Operational excellence", 0.92, "oee")

        # 6. Cycle Time Distribution
        if c["cycle_time"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["cycle_time"]].hist(ax=ax, bins=20, color="purple")
            ax.set_title("Cycle Time Variability")
            save(fig, "cycle_time_dist.png", "Process stability", 0.75, "efficiency")

        # 7. Output by Shift
        if c["shift"] and c["quantity"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["shift"])[c["quantity"]].mean().plot(kind="bar", ax=ax, color="teal")
            ax.set_title("Avg Output by Shift")
            save(fig, "shift_perf.png", "Shift comparison", 0.7, "workforce")

        # 8. Cost vs Defect Scatter
        if c["cost"] and c["defects"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df[c["defects"]], df[c["cost"]], alpha=0.5, color="brown")
            ax.set_title("Cost of Quality (Defects vs Cost)")
            ax.set_xlabel("Defect Qty")
            ax.set_ylabel("Production Cost")
            save(fig, "cost_quality.png", "Financial impact", 0.82, "financial")

        # Sort and Pick Top 4
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        defect = kpis.get("defect_rate", 0)
        oee = kpis.get("oee", 1.0)
        avail = kpis.get("availability", 1.0)
        perf = kpis.get("performance", 1.0)

        # Composite: "Hidden Factory" (High Output but High Defects)
        if perf > 0.95 and defect > 0.05:
            insights.append({
                "level": "CRITICAL", "title": "Hidden Factory Syndrome",
                "so_what": f"Speed is high (Performance {perf:.0%}) but Quality is suffering (Defects {defect:.1%}). Rework costs are likely high."
            })

        # Composite: "Downtime Crisis" (Low Availability kills OEE)
        if avail < 0.75 and oee < 0.60:
            insights.append({
                "level": "RISK", "title": "Equipment Reliability Crisis",
                "so_what": f"Availability is {avail:.1%}, driving OEE down to {oee:.1%}. Maintenance audit needed."
            })

        # Atomic Fallbacks
        if defect > 0.05 and not any("Hidden" in i["title"] for i in insights):
            insights.append({
                "level": "WARNING", "title": "Quality Alert",
                "so_what": f"Defect rate is {defect:.1%}, above 5% tolerance."
            })

        if oee < 0.65 and not any("Crisis" in i["title"] for i in insights):
            insights.append({
                "level": "WARNING", "title": "Low OEE",
                "so_what": f"Overall Equipment Effectiveness is {oee:.1%}. World class is 85%."
            })

        if not insights:
            insights.append({"level": "INFO", "title": "Production Stable", "so_what": "Manufacturing metrics healthy."})

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if "Hidden Factory Syndrome" in titles:
            recs.append({"action": "Slow down line speed to stabilize quality.", "priority": "HIGH"})
        
        if "Equipment Reliability Crisis" in titles:
            recs.append({"action": "Review preventative maintenance schedules.", "priority": "HIGH"})

        if kpis.get("defect_rate", 0) > 0.05:
            recs.append({"action": "Calibrate machines producing highest defects.", "priority": "MEDIUM"})

        return recs or [{"action": "Optimize cycle times.", "priority": "LOW"}]


# =====================================================
# DOMAIN DETECTOR
# =====================================================

class ManufacturingDomainDetector(BaseDomainDetector):
    domain_name = "manufacturing"
    TOKENS = {
        "production", "machine", "equipment", "downtime", "uptime", 
        "oee", "defect", "scrap", "yield", "cycle_time", "assembly", "factory"
    }

    def detect(self, df) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # Boost if Machine + Quantity exist
        cols = str(df.columns).lower()
        if "machine" in cols and ("qty" in cols or "production" in cols):
            confidence = max(confidence, 0.95)
            
        return DomainDetectionResult("manufacturing", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("manufacturing", ManufacturingDomain, ManufacturingDomainDetector)
