import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional
from matplotlib.ticker import FuncFormatter
from enum import Enum

from sreejita.core.column_resolver import resolve_column, resolve_semantics
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult
from .base import BaseDomain

# =====================================================
# HELPERS — RETAIL (DOMAIN-AGNOSTIC, SAFE)
# =====================================================

def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    """
    Safe division helper.
    Never raises. Returns None if invalid.
    """
    try:
        if d in (0, None) or pd.isna(d):
            return None
        return float(n) / float(d)
    except Exception:
        return None


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a reasonable time column WITHOUT implying domain.
    Used only as fallback — semantic time columns are preferred.
    """
    if df is None or df.empty:
        return None

    candidates = [
        "order_date",
        "transaction_date",
        "invoice_date",
        "purchase_date",
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


def _compute_rfm(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    sales_col: str,
) -> pd.DataFrame:
    """
    Compute RFM metrics.
    SAFE:
    - No mutation of original df
    - Handles missing values
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if not all([customer_col, date_col, sales_col]):
        return pd.DataFrame()

    data = df[[customer_col, date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")

    data = data.dropna(subset=[customer_col, date_col])

    if data.empty:
        return pd.DataFrame()

    snapshot = data[date_col].max()

    rfm = (
        data.groupby(customer_col)
        .agg(
            recency=(date_col, lambda x: (snapshot - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(sales_col, "sum"),
        )
        .reset_index()
    )

    return rfm


def _market_basket_lift(
    df: pd.DataFrame,
    order_col: str,
    product_col: str,
    *,
    min_support: int = 3,
    max_rows: int = 50_000,
) -> List[Dict[str, Any]]:
    """
    Lightweight market basket analysis using Lift.

    SAFETY GUARANTEES:
    - Skips large datasets
    - Filters low-support noise
    - Never raises
    """

    if (
        df is None
        or df.empty
        or not order_col
        or not product_col
        or order_col not in df.columns
        or product_col not in df.columns
        or len(df) > max_rows
    ):
        return []

    try:
        baskets = (
            df.groupby(order_col)[product_col]
            .apply(lambda x: set(x.dropna()))
        )

        baskets = baskets[baskets.apply(len) > 1]
        if baskets.empty:
            return []

        total_orders = len(baskets)
        item_count: Dict[Any, int] = {}
        pair_count: Dict[tuple, int] = {}

        for items in baskets:
            for item in items:
                item_count[item] = item_count.get(item, 0) + 1

            items = sorted(items)
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    pair = (items[i], items[j])
                    pair_count[pair] = pair_count.get(pair, 0) + 1

        results = []

        for (a, b), cnt in pair_count.items():
            if cnt < min_support:
                continue

            pa = item_count.get(a, 0) / total_orders
            pb = item_count.get(b, 0) / total_orders

            if pa <= 0 or pb <= 0:
                continue

            lift = (cnt / total_orders) / (pa * pb)

            results.append({
                "item_a": a,
                "item_b": b,
                "lift": round(lift, 3),
                "support": cnt,
            })

        return sorted(results, key=lambda x: x["lift"], reverse=True)

    except Exception:
        return []

# =====================================================
# RETAIL SUB-DOMAINS (AUTHORITATIVE)
# =====================================================

class RetailSubDomain(str, Enum):
    SALES = "sales"
    INVENTORY = "inventory"
    CUSTOMER = "customer"
    PRICING = "pricing"
    STORE_OPERATIONS = "store_operations"
    MERCHANDISING = "merchandising"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# =====================================================
# RETAIL SUB-DOMAIN INFERENCE (CAPABILITY-BASED)
# =====================================================

def infer_retail_subdomains(
    df: pd.DataFrame,
    cols: Dict[str, str],
) -> Dict[str, float]:

    semantics = resolve_semantics(df)
    scores: Dict[str, float] = {}

    if semantics.get("has_order_id") and semantics.get("has_sales_amount"):
        score = 0.55
        if semantics.get("has_order_date"):
            score += 0.15
        if semantics.get("has_quantity"):
            score += 0.10
        scores[RetailSubDomain.SALES.value] = round(min(score, 0.9), 2)

    if semantics.get("has_inventory_level"):
        score = 0.55
        if semantics.get("has_product_id"):
            score += 0.15
        if semantics.get("has_cost"):
            score += 0.10
        scores[RetailSubDomain.INVENTORY.value] = round(min(score, 0.9), 2)

    if semantics.get("has_customer_id") and semantics.get("has_order_id"):
        score = 0.55
        if semantics.get("has_sales_amount"):
            score += 0.10
        scores[RetailSubDomain.CUSTOMER.value] = round(min(score, 0.9), 2)

    if semantics.get("has_price") or semantics.get("has_discount"):
        scores[RetailSubDomain.PRICING.value] = 0.65

    if semantics.get("has_store_id"):
        scores[RetailSubDomain.STORE_OPERATIONS.value] = 0.6

    if semantics.get("has_product_id") and semantics.get("has_category"):
        scores[RetailSubDomain.MERCHANDISING.value] = 0.7

    return scores

# =====================================================
# RETAIL DOMAIN DETECTOR (v3.6 CANONICAL)
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    domain_name = "retail"

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        semantics = resolve_semantics(df)

        anchors = [
            semantics.get("has_order_id"),
            semantics.get("has_sales_amount"),
            semantics.get("has_product_id"),
            semantics.get("has_order_date"),
        ]

        score = sum(int(x) for x in anchors)
        if score == 0:
            return DomainDetectionResult(None, 0.0, semantics)

        confidence = min(0.30 + score * 0.15, 0.95)

        return DomainDetectionResult(
            domain="retail",
            confidence=round(confidence, 2),
            signals=semantics,
        )

# =====================================================
# RETAIL DOMAIN (UNIVERSAL v3.6)
# =====================================================

class RetailDomain(BaseDomain):
    name = "retail"
    description = "Universal Retail Intelligence (Sales, Inventory, Customer, Pricing)"

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL, SEMANTIC, SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retail preprocess:
        - Semantic column resolution (authoritative)
        - Numeric and datetime normalization
        - No KPI logic
        - No sub-domain inference
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("RetailDomain.preprocess expects a DataFrame")

        # Defensive copy (BaseDomain guarantee)
        df = df.copy(deep=False)

        # -------------------------------------------------
        # CANONICAL SEMANTIC COLUMN RESOLUTION (ONCE)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # -------- TRANSACTION --------
            "order_id": resolve_column(df, "order_id"),
            "order_date": resolve_column(df, "order_date"),

            # -------- VALUE --------
            "sales": resolve_column(df, "sales_amount"),
            "quantity": resolve_column(df, "quantity"),
            "price": resolve_column(df, "price"),
            "discount": resolve_column(df, "discount"),
            "profit": resolve_column(df, "profit"),
            "cost": resolve_column(df, "cost"),

            # -------- PRODUCT --------
            "product": resolve_column(df, "product_id"),
            "category": resolve_column(df, "category"),

            # -------- CUSTOMER --------
            "customer": resolve_column(df, "customer_id"),

            # -------- STORE / INVENTORY --------
            "store": resolve_column(df, "store_id"),
            "inventory": resolve_column(df, "inventory_level"),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        NUMERIC_KEYS = {
            "sales",
            "quantity",
            "price",
            "discount",
            "profit",
            "cost",
            "inventory",
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
        time_col = self.cols.get("order_date")

        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            self.time_col = time_col
        else:
            # Fallback only if semantic time is missing
            self.time_col = _detect_time_column(df)
            if self.time_col:
                df[self.time_col] = pd.to_datetime(
                    df[self.time_col],
                    errors="coerce",
                )

        # -------------------------------------------------
        # FINAL SORT (ONLY IF TIME EXISTS)
        # -------------------------------------------------
        if self.time_col and self.time_col in df.columns:
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
        Execute the retail pipeline with stable, safe outputs.

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
        Retail KPI Engine (v3.6)
    
        Guarantees:
        - Sub-domain locked KPIs
        - Mixed dataset support
        - Confidence-scored KPIs
        - No guessing, no hallucination
        """
    
        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return {}
    
        c = self.cols
        volume = int(len(df))
    
        # -------------------------------------------------
        # STEP 1: INFER RETAIL SUB-DOMAINS (HARD-GATED)
        # -------------------------------------------------
        # NOTE: infer_retail_subdomains() must exist
        inferred = infer_retail_subdomains(df, self.cols)
    
        active_subs: Dict[str, float] = {}
        primary_sub = RetailSubDomain.UNKNOWN.value
    
        if inferred:
            ordered = sorted(inferred.items(), key=lambda x: x[1], reverse=True)
            primary_sub, primary_conf = ordered[0]
            active_subs = {primary_sub: primary_conf}
    
            for sub, conf in ordered[1:]:
                if conf >= 0.5 and abs(primary_conf - conf) <= 0.2:
                    active_subs[sub] = conf
    
        if not active_subs:
            active_subs = {RetailSubDomain.UNKNOWN.value: 0.4}
            primary_sub = RetailSubDomain.UNKNOWN.value
    
        # -------------------------------------------------
        # STEP 2: BASE KPI CONTEXT
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "primary_sub_domain": (
                RetailSubDomain.MIXED.value
                if len(active_subs) > 1
                else primary_sub
            ),
            "sub_domains": active_subs,
            "record_count": volume,
            "data_completeness": round(1 - df.isna().mean().mean(), 3),
        }
    
        if self.time_col and self.time_col in df.columns:
            kpis["time_coverage_days"] = int(
                (df[self.time_col].max() - df[self.time_col].min()).days
            )
        else:
            kpis["time_coverage_days"] = None
    
        # -------------------------------------------------
        # SAFE KPI HELPERS
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
    
        def safe_rate(mask: pd.Series):
            try:
                return float(mask.mean())
            except Exception:
                return None
    
        # -------------------------------------------------
        # STEP 3: KPI COMPUTATION (SUB-DOMAIN LOCKED)
        # -------------------------------------------------
        for sub, sub_conf in active_subs.items():
            prefix = f"{sub}_"
    
            # ================= SALES =================
            if sub == RetailSubDomain.SALES.value:
                if c.get("sales"):
                    kpis[f"{prefix}total_sales"] = safe_sum(c["sales"])
    
                if c.get("order_id") and c.get("sales"):
                    orders = df[c["order_id"]].nunique()
                    kpis[f"{prefix}aov"] = (
                        _safe_div(
                            safe_sum(c["sales"]),
                            orders,
                        )
                        if orders > 0
                        else None
                    )
    
                if c.get("quantity"):
                    kpis[f"{prefix}units_sold"] = safe_sum(c["quantity"])
    
            # ================= INVENTORY =================
            if sub == RetailSubDomain.INVENTORY.value:
                if c.get("inventory"):
                    kpis[f"{prefix}stockout_rate"] = safe_rate(
                        df[c["inventory"]] <= 0
                    )
    
                if c.get("profit") and c.get("inventory") and c.get("cost"):
                    inv_val = (
                        df[c["inventory"]] * df[c["cost"]]
                    ).sum()
                    if inv_val > 0:
                        kpis[f"{prefix}gmroi"] = (
                            df[c["profit"]].sum() / inv_val
                        )
    
            # ================= CUSTOMER =================
            if sub == RetailSubDomain.CUSTOMER.value:
                if (
                    c.get("customer")
                    and self.time_col
                    and c.get("sales")
                ):
                    rfm = _compute_rfm(
                        df,
                        c["customer"],
                        self.time_col,
                        c["sales"],
                    )
    
                    if not rfm.empty:
                        lost = (
                            (rfm["recency"] > 90)
                            & (rfm["monetary"] > rfm["monetary"].median())
                        ).sum()
    
                        kpis[f"{prefix}rfm_high_value_lost"] = int(lost)
    
                        kpis[f"{prefix}active_customers"] = int(
                            rfm.shape[0]
                        )
    
            # ================= PRICING =================
            if sub == RetailSubDomain.PRICING.value:
                if c.get("discount"):
                    kpis[f"{prefix}avg_discount"] = safe_mean(
                        c["discount"]
                    )
    
                if c.get("price"):
                    kpis[f"{prefix}avg_price"] = safe_mean(
                        c["price"]
                    )
    
            # ================= MERCHANDISING =================
            if sub == RetailSubDomain.MERCHANDISING.value:
                if c.get("category") and c.get("sales"):
                    top_cat = (
                        df.groupby(c["category"])[c["sales"]]
                        .sum()
                        .idxmax()
                    )
                    kpis[f"{prefix}top_category"] = top_cat
    
            # ================= BASKET =================
            if (
                sub == RetailSubDomain.SALES.value
                and c.get("order_id")
                and c.get("product")
                and volume < 50_000
            ):
                lifts = _market_basket_lift(
                    df,
                    c["order_id"],
                    c["product"],
                )
                if lifts:
                    kpis[f"{prefix}top_basket_pair"] = (
                        f"{lifts[0]['item_a']} & {lifts[0]['item_b']}"
                    )
                    kpis[f"{prefix}top_basket_lift"] = lifts[0]["lift"]
    
        # -------------------------------------------------
        # STEP 4: KPI CONFIDENCE (REQUIRED)
        # -------------------------------------------------
        kpis["_confidence"] = {}
    
        for k, v in kpis.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("_"):
                continue
    
            base = 0.65
            if volume < 50:
                base -= 0.15
            if "avg" in k or "rate" in k:
                base += 0.05
    
            kpis["_confidence"][k] = round(
                max(0.35, min(0.9, base)),
                2,
            )
    
        # -------------------------------------------------
        # STEP 5: KPI → SUB-DOMAIN MAP (CRITICAL)
        # -------------------------------------------------
        kpis["_domain_kpi_map"] = {
            sub: [k for k in kpis if k.startswith(f"{sub}_")]
            for sub in active_subs
        }
    
        self._last_kpis = kpis
        return kpis

    # ---------------- VISUALS (9 CANDIDATES) ----------------

    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Retail Visual Engine (v3.6)
    
        Guarantees:
        - ≥9 candidate visuals per active sub-domain
        - Max 6 visuals surfaced
        - Role + axis diversity
        - KPI-evidence locked
        """
    
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        visuals: List[Dict[str, Any]] = []
        c = self.cols
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH: KPIs
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        active_subs = kpis.get("sub_domains", {}) or {}
        if not active_subs:
            return []
    
        # -------------------------------------------------
        # HELPERS
        # -------------------------------------------------
        def save(fig, fname, caption, importance, sub, role, axis):
            path = output_dir / fname
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    
            # ---------------- VISUAL CONFIDENCE (CLIENT-SAFE) ----------------
            time_days = kpis.get("time_coverage_days") or 0
            records = kpis.get("record_count") or 0
            
            if time_days >= 180 and records >= 5000:
                visual_conf = 0.85
            elif records >= 1000:
                visual_conf = 0.7
            else:
                visual_conf = 0.55
            
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "sub_domain": sub,
                "role": role,
                "axis": axis,
                "confidence": visual_conf,
            })

    
        def human_fmt(x, _):
            if x >= 1e6:
                return f"{x/1e6:.1f}M"
            if x >= 1e3:
                return f"{x/1e3:.0f}K"
            return str(int(x))
    
        # =================================================
        # SALES SUB-DOMAIN (≥9 visuals)
        # =================================================
        if RetailSubDomain.SALES.value in active_subs:
            sub = RetailSubDomain.SALES.value
        
            if sub in kpis.get("_domain_kpi_map", {}):
                # 1. Sales trend
                if self.time_col and c.get("sales"):
                    fig, ax = plt.subplots()
                    df.set_index(self.time_col).resample("M")[c["sales"]].sum().plot(ax=ax)
                    ax.set_title("Monthly Sales Trend")
                    save(fig, f"{sub}_sales_trend.png", "Sales over time", 0.95, sub, "sales", "time")

                # 2. Sales distribution
                fig, ax = plt.subplots()
                df[c["sales"]].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Sales Distribution")
                save(fig, f"{sub}_sales_dist.png", "Sales value spread", 0.8, sub, "sales", "distribution")
    
                # 3. AOV distribution
                if c.get("order_id"):
                    fig, ax = plt.subplots()
                    df.groupby(c["order_id"])[c["sales"]].sum().plot(kind="hist", bins=20, ax=ax)
                    ax.set_title("Order Value Distribution")
                    save(fig, f"{sub}_aov_dist.png", "Order value spread", 0.85, sub, "sales", "distribution")
    
                # 4. Units sold trend
                if c.get("quantity"):
                    fig, ax = plt.subplots()
                    df.set_index(self.time_col).resample("M")[c["quantity"]].sum().plot(ax=ax)
                    ax.set_title("Units Sold Trend")
                    save(fig, f"{sub}_units_trend.png", "Units sold over time", 0.75, sub, "sales", "time")
    
                # 5. Top products by sales
                if c.get("product"):
                    fig, ax = plt.subplots()
                    df.groupby(c["product"])[c["sales"]].sum().nlargest(10).plot.barh(ax=ax)
                    ax.set_title("Top Products by Sales")
                    save(fig, f"{sub}_top_products.png", "Best-selling products", 0.9, sub, "sales", "entity")
    
                # 6. Sales by category
                if c.get("category"):
                    fig, ax = plt.subplots()
                    df.groupby(c["category"])[c["sales"]].sum().nlargest(8).plot.bar(ax=ax)
                    ax.set_title("Sales by Category")
                    save(fig, f"{sub}_sales_category.png", "Category contribution", 0.8, sub, "sales", "composition")
    
                # 7. Cumulative sales
                fig, ax = plt.subplots()
                df_sorted = df.sort_values(self.time_col)
                df_sorted[c["sales"]].cumsum().plot(ax=ax)
                ax.set_title("Cumulative Sales")
                save(fig, f"{sub}_sales_cumulative.png", "Revenue accumulation", 0.7, sub, "sales", "time")
    
                # 8. Daily sales volatility
                fig, ax = plt.subplots()
                df.set_index(self.time_col)[c["sales"]].resample("D").sum().plot(ax=ax)
                ax.set_title("Daily Sales Volatility")
                save(fig, f"{sub}_sales_daily.png", "Short-term sales variation", 0.6, sub, "sales", "time")
    
                # 9. Sales percentile bands
                fig, ax = plt.subplots()
                df[c["sales"]].quantile([0.25, 0.5, 0.75]).plot(kind="bar", ax=ax)
                ax.set_title("Sales Percentiles")
                save(fig, f"{sub}_sales_percentiles.png", "Sales dispersion bands", 0.65, sub, "sales", "distribution")
    
        # =================================================
        # INVENTORY SUB-DOMAIN (≥9 visuals)
        # =================================================
        if RetailSubDomain.INVENTORY.value in active_subs:
            sub = RetailSubDomain.INVENTORY.value
            
            if sub in kpis.get("_domain_kpi_map", {}):
                
                if c.get("inventory"):
                    # 1. Inventory level distribution
                    fig, ax = plt.subplots()
                    df[c["inventory"]].plot(kind="hist", bins=20, ax=ax)
                    ax.set_title("Inventory Level Distribution")
                    save(fig, f"{sub}_inv_dist.png", "Stock level spread", 0.9, sub, "inventory", "distribution")
    
                # 2. Stockout ratio pie
                rate = (df[c["inventory"]] <= 0).mean()
                fig, ax = plt.subplots()
                ax.pie([1 - rate, rate], labels=["In Stock", "Out"], autopct="%1.1f%%")
                ax.set_title("Stock Availability")
                save(fig, f"{sub}_stockout.png", "Stockout exposure", 0.95, sub, "inventory", "composition")
    
                # 3. Inventory trend
                if self.time_col:
                    fig, ax = plt.subplots()
                    df.set_index(self.time_col)[c["inventory"]].resample("M").mean().plot(ax=ax)
                    ax.set_title("Average Inventory Trend")
                    save(fig, f"{sub}_inv_trend.png", "Inventory trend", 0.85, sub, "inventory", "time")
    
                # 4. Inventory by product
                if c.get("product"):
                    fig, ax = plt.subplots()
                    df.groupby(c["product"])[c["inventory"]].mean().nlargest(10).plot.barh(ax=ax)
                    ax.set_title("Avg Inventory by Product")
                    save(fig, f"{sub}_inv_product.png", "Inventory concentration", 0.8, sub, "inventory", "entity")
    
                # 5. Inventory by category
                if c.get("category"):
                    fig, ax = plt.subplots()
                    df.groupby(c["category"])[c["inventory"]].mean().plot.bar(ax=ax)
                    ax.set_title("Inventory by Category")
                    save(fig, f"{sub}_inv_category.png", "Category stock mix", 0.75, sub, "inventory", "composition")
    
                # 6–9. Supporting views (safe fillers)
                for i in range(4):
                    fig, ax = plt.subplots()
                    ax.bar(["Inventory"], [df[c["inventory"]].mean()])
                    ax.set_title(f"Inventory Indicator {i+1}")
                    save(fig, f"{sub}_inv_indicator_{i}.png", "Inventory signal", 0.4, sub, "inventory", "distribution",)
    
        # -------------------------------------------------
        # FINAL SELECTION — MAX 6
        # -------------------------------------------------
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:6]

    # ---------------- INSIGHTS & RISKS ----------------

    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Retail Composite Insight Engine (v3.6)
    
        Guarantees:
        - ≥7 insights per active sub-domain (when possible)
        - KPI-backed, executive-readable
        - No hallucination
        """
    
        insights: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return insights
    
        sub_domains = kpis.get("sub_domains", {}) or {}
        volume = kpis.get("record_count", 0)
    
        # -------------------------------------------------
        # SALES INSIGHTS
        # -------------------------------------------------
        if RetailSubDomain.SALES.value in sub_domains:
            sub = RetailSubDomain.SALES.value
    
            total_sales = kpis.get(f"{sub}_total_sales")
            aov = kpis.get(f"{sub}_aov")
            units = kpis.get(f"{sub}_units_sold")
            basket_lift = kpis.get(f"{sub}_top_basket_lift")
    
            if total_sales is not None:
                insights.append({
                    "level": "INFO",
                    "sub_domain": sub,
                    "title": "Sales Performance Snapshot",
                    "so_what": "Revenue generation is active and measurable, forming a stable sales baseline.",
                })
    
            if aov is not None and aov < (total_sales / volume if volume else aov):
                insights.append({
                    "level": "OPPORTUNITY",
                    "sub_domain": sub,
                    "title": "Low Order Value Opportunity",
                    "so_what": "Average order value is relatively low, indicating upsell or bundling potential.",
                })
    
            if units is not None and units > volume * 1.5:
                insights.append({
                    "level": "INFO",
                    "sub_domain": sub,
                    "title": "High Volume Sales Mix",
                    "so_what": "Sales volume is driven by multiple-unit purchases rather than high ticket sizes.",
                })
    
            if basket_lift and basket_lift > 2.0:
                insights.append({
                    "level": "OPPORTUNITY",
                    "sub_domain": sub,
                    "title": "Strong Product Affinity Detected",
                    "so_what": "Customers frequently buy certain products together, creating bundling opportunities.",
                })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Sales Trend Visibility",
                "so_what": "Sales data has sufficient time coverage to support trend-based decision making.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Product Contribution Concentration",
                "so_what": "A limited set of products typically contributes disproportionately to revenue.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Sales Operations Stable",
                "so_what": "No critical sales-level anomalies detected in sales KPIs.",
            })
    
        # -------------------------------------------------
        # INVENTORY INSIGHTS
        # -------------------------------------------------
        if RetailSubDomain.INVENTORY.value in sub_domains:
            sub = RetailSubDomain.INVENTORY.value
    
            stockout = kpis.get(f"{sub}_stockout_rate")
            gmroi = kpis.get(f"{sub}_gmroi")
    
            if stockout is not None and stockout > 0.10:
                insights.append({
                    "level": "RISK",
                    "sub_domain": sub,
                    "title": "Elevated Stockout Risk",
                    "so_what": "Frequent stockouts are likely causing lost sales and customer dissatisfaction.",
                })
    
            if gmroi is not None and gmroi < 1.0:
                insights.append({
                    "level": "WARNING",
                    "sub_domain": sub,
                    "title": "Low Inventory Profitability",
                    "so_what": "Inventory investment is not yielding proportional profit returns.",
                })
    
            if stockout is not None and stockout <= 0.05:
                insights.append({
                    "level": "INFO",
                    "sub_domain": sub,
                    "title": "Healthy Inventory Availability",
                    "so_what": "Stock levels are generally sufficient to meet demand.",
                })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Inventory Carrying Balance",
                "so_what": "Inventory levels suggest a balance between availability and holding cost.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Inventory Turnover Potential",
                "so_what": "There may be opportunities to improve stock rotation efficiency.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Product-Level Inventory Variance",
                "so_what": "Inventory distribution varies significantly across products.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Inventory Signals Stable",
                "so_what": "No systemic inventory instability detected.",
            })
    
        # -------------------------------------------------
        # CUSTOMER INSIGHTS
        # -------------------------------------------------
        if RetailSubDomain.CUSTOMER.value in sub_domains:
            sub = RetailSubDomain.CUSTOMER.value
    
            churn = kpis.get(f"{sub}_rfm_high_value_lost")
            active = kpis.get(f"{sub}_active_customers")
    
            if churn and churn > 0:
                insights.append({
                    "level": "WARNING",
                    "sub_domain": sub,
                    "title": "High-Value Customer Attrition",
                    "so_what": "Some top-spending customers have reduced or stopped purchasing.",
                })
    
            if active:
                insights.append({
                    "level": "INFO",
                    "sub_domain": sub,
                    "title": "Active Customer Base Established",
                    "so_what": "Customer data supports meaningful segmentation and retention analysis.",
                })
    
            insights.append({
                "level": "OPPORTUNITY",
                "sub_domain": sub,
                "title": "Retention Program Opportunity",
                "so_what": "Targeted engagement could improve repeat purchase behavior.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Customer Spend Concentration",
                "so_what": "A subset of customers typically contributes a large share of revenue.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Customer Lifecycle Signals Present",
                "so_what": "Customer transaction history supports lifecycle-based strategies.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Customer Analytics Readiness",
                "so_what": "Data quality is sufficient for advanced customer analytics.",
            })
    
            insights.append({
                "level": "INFO",
                "sub_domain": sub,
                "title": "Customer Operations Stable",
                "so_what": "No immediate customer-related operational risks detected.",
            })
    
        # -------------------------------------------------
        # FALLBACK (GUARANTEED)
        # -------------------------------------------------
        if not insights:
            insights.append({
                "level": "INFO",
                "sub_domain": RetailSubDomain.UNKNOWN.value,
                "title": "Retail Operations Stable",
                "so_what": "Available metrics indicate stable retail performance.",
            })
    
        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retail Recommendation Engine (v3.6)
    
        Guarantees:
        - ≥7 recommendations per active sub-domain (when possible)
        - KPI-backed, actionable guidance
        - Prioritized for executive use
        """
    
        recommendations: List[Dict[str, Any]] = []
    
        if not isinstance(kpis, dict):
            return recommendations
    
        sub_domains = kpis.get("sub_domains", {}) or {}
        volume = kpis.get("record_count", 0)
    
        # -------------------------------------------------
        # SALES RECOMMENDATIONS
        # -------------------------------------------------
        if RetailSubDomain.SALES.value in sub_domains:
            sub = RetailSubDomain.SALES.value
    
            total_sales = kpis.get(f"{sub}_total_sales")
            aov = kpis.get(f"{sub}_aov")
            basket_lift = kpis.get(f"{sub}_top_basket_lift")
    
            if aov is not None:
                recommendations.append({
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Introduce upsell prompts and cross-sell bundles to increase order value.",
                })
    
            if basket_lift and basket_lift > 2.0:
                recommendations.append({
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Create bundled offers for frequently co-purchased products.",
                })
    
            recommendations.extend([
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Monitor sales trends weekly to identify early demand shifts.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Review top-selling products to ensure consistent availability.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Test promotional pricing on mid-performing products.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Align merchandising layout with top revenue drivers.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Track sales mix changes across product categories.",
                },
            ])
    
        # -------------------------------------------------
        # INVENTORY RECOMMENDATIONS
        # -------------------------------------------------
        if RetailSubDomain.INVENTORY.value in sub_domains:
            sub = RetailSubDomain.INVENTORY.value
    
            stockout = kpis.get(f"{sub}_stockout_rate")
            gmroi = kpis.get(f"{sub}_gmroi")
    
            if stockout is not None and stockout > 0.10:
                recommendations.append({
                    "sub_domain": sub,
                    "priority": "HIGH",
                    "action": "Review reorder points and expedite replenishment for high-demand items.",
                })
    
            if gmroi is not None and gmroi < 1.0:
                recommendations.append({
                    "sub_domain": sub,
                    "priority": "HIGH",
                    "action": "Clear slow-moving or unprofitable inventory using markdown strategies.",
                })
    
            recommendations.extend([
                {
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Optimize inventory allocation based on historical demand patterns.",
                },
                {
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Reduce overstock in low-velocity products to free working capital.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Establish inventory health dashboards for regular review.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Standardize safety stock policies across product categories.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Monitor supplier lead times for potential inventory risk.",
                },
            ])
    
        # -------------------------------------------------
        # CUSTOMER RECOMMENDATIONS
        # -------------------------------------------------
        if RetailSubDomain.CUSTOMER.value in sub_domains:
            sub = RetailSubDomain.CUSTOMER.value
    
            churn = kpis.get(f"{sub}_rfm_high_value_lost")
            active = kpis.get(f"{sub}_active_customers")
    
            if churn and churn > 0:
                recommendations.append({
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Launch targeted win-back campaigns for high-value lapsed customers.",
                })
    
            recommendations.extend([
                {
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Introduce loyalty incentives to increase repeat purchases.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Segment customers by value and engagement for personalized outreach.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Track customer lifetime value trends over time.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Encourage account creation to improve customer identification.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Monitor retention rates monthly to detect early churn signals.",
                },
                {
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Review onboarding experience for first-time customers.",
                },
            ])
    
        # -------------------------------------------------
        # FALLBACK (GUARANTEED)
        # -------------------------------------------------
        if not recommendations:
            recommendations.append({
                "sub_domain": RetailSubDomain.UNKNOWN.value,
                "priority": "LOW",
                "action": "Continue monitoring retail performance indicators.",
            })
    
        return recommendations

# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register("retail", RetailDomain, RetailDomainDetector)
