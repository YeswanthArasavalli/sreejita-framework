import pandas as pd
import numpy as np
import matplotlib

# Headless backend for CI / server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional
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

    Guarantees:
    - Never raises
    - Never invents values
    - Returns None on invalid input
    """
    try:
        if d in (0, None) or pd.isna(d):
            return None
        if n is None or pd.isna(n):
            return None
        return float(n) / float(d)
    except Exception:
        return None


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a reasonable time column WITHOUT implying domain meaning.

    Notes:
    - Used strictly as a fallback
    - Semantic time resolution is always preferred
    - Never mutates data
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
        if any(key in col_l for key in candidates):
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
    Compute RFM metrics (INTERNAL USE ONLY).

    Guarantees:
    - No mutation of original DataFrame
    - Safe under missing or malformed data
    - Returns empty DataFrame on insufficient evidence

    IMPORTANT:
    - Outputs are NOT KPIs
    - Outputs must never be exposed directly
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if not customer_col or not date_col or not sales_col:
        return pd.DataFrame()

    if customer_col not in df.columns or date_col not in df.columns:
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
    Lightweight market basket analysis (Lift metric).

    SAFETY GUARANTEES:
    - Skips large datasets
    - Filters low-support noise
    - Never raises exceptions
    - Returns empty list when evidence is weak

    IMPORTANT:
    - Results are descriptive only
    - Must NOT be treated as KPIs
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

        results: List[Dict[str, Any]] = []

        for (a, b), cnt in pair_count.items():
            if cnt < min_support:
                continue

            pa = item_count.get(a, 0) / total_orders
            pb = item_count.get(b, 0) / total_orders

            if pa <= 0 or pb <= 0:
                continue

            lift = (cnt / total_orders) / (pa * pb)

            results.append(
                {
                    "item_a": a,
                    "item_b": b,
                    "lift": round(lift, 3),
                    "support": cnt,
                }
            )

        return sorted(results, key=lambda x: x["lift"], reverse=True)

    except Exception:
        return []

# =====================================================
# RETAIL SUB-DOMAINS (AUTHORITATIVE ENUM)
# =====================================================

class RetailSubDomain(str, Enum):
    """
    Canonical retail sub-domains.

    Notes:
    - Used for capability routing only
    - NOT KPIs
    - NOT confidence-bearing
    """

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
    """
    Infer active retail sub-domains based on semantic capabilities.

    Guarantees:
    - Deterministic
    - No confidence inflation
    - No hard dependency on column names
    - Returns EMPTY dict when evidence is weak

    IMPORTANT:
    - Scores are routing weights, NOT confidence
    - Output must never be exposed directly to users
    """

    if df is None or df.empty:
        return {}

    semantics = resolve_semantics(df)
    scores: Dict[str, float] = {}

    # ---------------- SALES ----------------
    if semantics.get("has_order_id") and semantics.get("has_sales_amount"):
        score = 0.55
        if semantics.get("has_order_date"):
            score += 0.15
        if semantics.get("has_quantity"):
            score += 0.10
        scores[RetailSubDomain.SALES.value] = round(min(score, 0.9), 2)

    # ---------------- INVENTORY ----------------
    if semantics.get("has_inventory_level"):
        score = 0.55
        if semantics.get("has_product_id"):
            score += 0.15
        if semantics.get("has_cost"):
            score += 0.10
        scores[RetailSubDomain.INVENTORY.value] = round(min(score, 0.9), 2)

    # ---------------- CUSTOMER ----------------
    if semantics.get("has_customer_id") and semantics.get("has_order_id"):
        score = 0.55
        if semantics.get("has_sales_amount"):
            score += 0.10
        scores[RetailSubDomain.CUSTOMER.value] = round(min(score, 0.9), 2)

    # ---------------- PRICING ----------------
    if semantics.get("has_price") or semantics.get("has_discount"):
        scores[RetailSubDomain.PRICING.value] = 0.65

    # ---------------- STORE OPERATIONS ----------------
    if semantics.get("has_store_id"):
        scores[RetailSubDomain.STORE_OPERATIONS.value] = 0.6

    # ---------------- MERCHANDISING ----------------
    if semantics.get("has_product_id") and semantics.get("has_category"):
        scores[RetailSubDomain.MERCHANDISING.value] = 0.7

    return scores


# =====================================================
# RETAIL DOMAIN DETECTOR (PHASE-5 SAFE)
# =====================================================

class RetailDomainDetector(BaseDomainDetector):
    """
    Retail domain detector.

    Guarantees:
    - Conservative confidence
    - Capability-based detection
    - No over-assertion under weak evidence
    """

    domain_name = "retail"

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        if df is None or df.empty:
            return DomainDetectionResult(None, 0.0, {})

        semantics = resolve_semantics(df)

        anchors = [
            bool(semantics.get("has_order_id")),
            bool(semantics.get("has_sales_amount")),
            bool(semantics.get("has_product_id")),
            bool(semantics.get("has_order_date")),
        ]

        score = sum(anchors)
        if score == 0:
            return DomainDetectionResult(None, 0.0, semantics)

        # Conservative, monotonic confidence (Phase-5 aligned)
        confidence = min(0.30 + score * 0.10, 0.80)

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
        numeric_keys = {
            "sales",
            "quantity",
            "price",
            "discount",
            "profit",
            "cost",
            "inventory",
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
        # DATETIME NORMALIZATION
        # -------------------------------------------------
        self.time_col = None
        time_col = self.cols.get("order_date")

        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            self.time_col = time_col
        else:
            # Fallback only if semantic time is missing
            fallback = _detect_time_column(df)
            if fallback and fallback in df.columns:
                df[fallback] = pd.to_datetime(df[fallback], errors="coerce")
                self.time_col = fallback

        # -------------------------------------------------
        # FINAL SORT (ONLY IF TIME EXISTS)
        # -------------------------------------------------
        if self.time_col and self.time_col in df.columns:
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
        Execute the retail pipeline with Phase-5 safety guarantees.

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

            # Infer sub-domains ONLY (no KPI exposure)
            self.calculate_kpis(df)

            # Visuals are allowed, but must not infer confidence
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
        Retail KPI stub (Phase 5 – trust-correction).

        Guarantees:
        - NO KPI invention
        - NO inline KPI math exposed
        - NO confidence creation or mutation
        - Sub-domain inference ONLY (internal routing)
        - Safe under weak or empty data

        NOTE:
        - Retail KPIs are not yet registry-approved
        - This method intentionally returns an EMPTY dict
        """

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            self._active_subdomains = {}
            return {}

        # -------------------------------------------------
        # SUB-DOMAIN INFERENCE (ROUTING ONLY)
        # -------------------------------------------------
        inferred = infer_retail_subdomains(df, self.cols)

        active_subs: Dict[str, float] = {}

        if inferred:
            ordered = sorted(inferred.items(), key=lambda x: x[1], reverse=True)
            primary_sub, primary_conf = ordered[0]
            active_subs = {primary_sub: primary_conf}

            for sub, conf in ordered[1:]:
                if conf >= 0.5 and abs(primary_conf - conf) <= 0.2:
                    active_subs[sub] = conf

        # Never fabricate UNKNOWN with fake confidence
        self._active_subdomains = active_subs or {}

        # -------------------------------------------------
        # PHASE 5 RULE:
        # Do NOT expose KPIs from retail yet
        # -------------------------------------------------
        return {}


    # ---------------- VISUALS (9 CANDIDATES) ----------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Retail Visual Engine (Phase 5 – descriptive only)

        Guarantees:
        - Deterministic
        - No KPI dependency
        - No confidence invention
        - No narrative completeness assumptions
        - Safe under weak data
        """

        if df is None or df.empty:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        visuals: List[Dict[str, Any]] = []
        c = self.cols

        # -------------------------------------------------
        # ACTIVE SUB-DOMAINS (ROUTING ONLY)
        # -------------------------------------------------
        active_subs = getattr(self, "_active_subdomains", {}) or {}
        if not active_subs:
            return []

        # -------------------------------------------------
        # HELPER: SAFE SAVE
        # -------------------------------------------------
        def save(fig, fname, caption, importance, sub, role, axis):
            path = output_dir / fname
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            visuals.append(
                {
                    "path": str(path),
                    "caption": caption,
                    "importance": float(importance),
                    "sub_domain": sub,
                    "role": role,
                    "axis": axis,
                    # Phase-5 rule: visuals do NOT infer confidence
                    "confidence": None,
                }
            )

        # =================================================
        # SALES VISUALS (DESCRIPTIVE ONLY)
        # =================================================
        if RetailSubDomain.SALES.value in active_subs:
            sub = RetailSubDomain.SALES.value

            if self.time_col and c.get("sales"):
                # Sales over time
                fig, ax = plt.subplots()
                df.set_index(self.time_col).resample("M")[c["sales"]].sum().plot(ax=ax)
                ax.set_title("Monthly Sales Trend")
                save(fig, f"{sub}_sales_trend.png", "Sales over time", 0.9, sub, "sales", "time")

            if c.get("sales"):
                # Sales distribution
                fig, ax = plt.subplots()
                df[c["sales"]].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Sales Distribution")
                save(fig, f"{sub}_sales_dist.png", "Sales value distribution", 0.7, sub, "sales", "distribution")

            if c.get("order_id") and c.get("sales"):
                # Order value distribution
                fig, ax = plt.subplots()
                df.groupby(c["order_id"])[c["sales"]].sum().plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Order Value Distribution")
                save(fig, f"{sub}_order_value_dist.png", "Order value distribution", 0.75, sub, "sales", "distribution")

            if c.get("product") and c.get("sales"):
                # Top products by sales
                fig, ax = plt.subplots()
                df.groupby(c["product"])[c["sales"]].sum().nlargest(10).plot.barh(ax=ax)
                ax.set_title("Top Products by Sales")
                save(fig, f"{sub}_top_products.png", "Top products by sales", 0.8, sub, "sales", "entity")

        # =================================================
        # INVENTORY VISUALS (DESCRIPTIVE ONLY)
        # =================================================
        if RetailSubDomain.INVENTORY.value in active_subs:
            sub = RetailSubDomain.INVENTORY.value

            if c.get("inventory"):
                # Inventory distribution
                fig, ax = plt.subplots()
                df[c["inventory"]].dropna().plot(kind="hist", bins=20, ax=ax)
                ax.set_title("Inventory Level Distribution")
                save(fig, f"{sub}_inventory_dist.png", "Inventory level distribution", 0.8, sub, "inventory", "distribution")

            if self.time_col and c.get("inventory"):
                # Inventory trend
                fig, ax = plt.subplots()
                df.set_index(self.time_col)[c["inventory"]].resample("M").mean().plot(ax=ax)
                ax.set_title("Average Inventory Trend")
                save(fig, f"{sub}_inventory_trend.png", "Inventory over time", 0.75, sub, "inventory", "time")

            if c.get("category") and c.get("inventory"):
                # Inventory by category
                fig, ax = plt.subplots()
                df.groupby(c["category"])[c["inventory"]].mean().plot.bar(ax=ax)
                ax.set_title("Inventory by Category")
                save(fig, f"{sub}_inventory_category.png", "Inventory by category", 0.7, sub, "inventory", "composition")

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
        Retail Insight Engine (Phase 5 – intentionally silent).

        Guarantees:
        - No insight generation without approved KPIs
        - No narrative under weak or missing evidence
        - No confidence inflation
        - Stable, deterministic behavior

        NOTE:
        - Retail KPIs are not yet registry-approved
        - Therefore, insights are intentionally suppressed
        """

        # Phase 5 rule:
        # Silence is better than false certainty
        return []

    # ---------------- RECOMMENDATIONS ----------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retail Recommendation Engine (Phase 5 – intentionally silent).

        Guarantees:
        - No recommendations without approved KPIs
        - No generic or speculative advice
        - No confidence inheritance without evidence
        - Deterministic, stable behavior

        NOTE:
        - Retail KPIs are not yet registry-approved
        - Insights are intentionally suppressed
        - Therefore, recommendations MUST be empty
        """

        # Phase 5 rule:
        # Recommendations require strong, explicit evidence.
        # In its absence, silence is the only safe behavior.
        return []


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register("retail", RetailDomain, RetailDomainDetector)
