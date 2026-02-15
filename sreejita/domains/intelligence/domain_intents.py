# =====================================================
# DOMAIN INTENTS — CANONICAL & NORMALIZATION-ALIGNED
# Sreejita Framework v3.6 (FINAL)
# =====================================================

"""
PURPOSE:
- Provide weak, vocabulary-level intent signals
- Support intelligence layers (intent scoring, explainability)
- NEVER override detectors
- NEVER infer domain ownership directly

DESIGN RULES:
- Tokens MUST already be normalized (snake_case)
- Tokens are SEMANTIC HINTS, not rules
- Used ONLY as soft priors
"""

DOMAIN_INTENTS = {

    # =====================================================
    # SUPPLY CHAIN 🚚
    # =====================================================
    "supply_chain": {
        "high": {
            # Inventory & warehousing
            "warehouse", "inventory", "inventory_level",
            "stock", "stock_on_hand", "safety_stock",
            "reorder_point", "backorder",

            # Logistics & movement
            "carrier", "freight", "shipping_mode",
            "delivery_status", "tracking_number",
            "route", "shipment", "ship_date",
            "delivery_date", "promised_date",

            # Procurement
            "supplier", "vendor", "procurement",

            # Flow metrics
            "lead_time", "cycle_time",
            "on_time_delivery", "fill_rate",

            # Product
            "sku"
        },
        "ambiguous": {
            "order_id", "quantity", "location", "date", "status"
        }
    },

    # =====================================================
    # HR 👥
    # =====================================================
    "hr": {
        "high": {
            # Identity
            "employee_id", "employee_name", "staff_id",

            # Organization
            "department", "designation", "role", "manager",

            # Compensation
            "salary", "compensation", "ctc", "payroll", "bonus",

            # Lifecycle
            "attrition", "termination", "resignation",
            "hire_date", "joining_date", "exit_date",

            # Performance & attendance
            "performance_score", "rating",
            "leave_balance", "attendance", "timesheet",
            "absence", "tenure"
        },
        "ambiguous": {
            "id", "date", "gender", "age", "location", "status"
        }
    },

    # =====================================================
    # MARKETING 📢
    # =====================================================
    "marketing": {
        "high": {
            # Campaign structure
            "campaign_id", "ad_group", "creative_id",

            # Reach & engagement
            "impressions", "clicks", "ctr",

            # Cost & efficiency
            "cpc", "cpm", "ad_spend",
            "cost_per_acquisition", "roas",

            # Attribution
            "utm_source", "utm_medium", "utm_campaign"
        },
        "ambiguous": {
            "channel", "cost", "revenue", "date"
        }
    },

    # =====================================================
    # RETAIL 🛍️
    # =====================================================
    "retail": {
        "high": {
            # Store & POS
            "store_id", "store_location", "pos_id",
            "cashier",

            # Merchandising
            "shelf", "aisle", "zone",

            # Promotions
            "promotion", "markdown",

            # Customer behavior
            "loyalty_card", "foot_traffic",
            "basket_size"
        },
        "ambiguous": {
            "sales", "quantity", "price", "sku", "date"
        }
    },

    # =====================================================
    # ECOMMERCE 🛒
    # =====================================================
    "ecommerce": {
        "high": {
            # Funnel
            "cart_abandonment", "add_to_cart",
            "checkout", "conversion_rate",

            # Economics
            "aov", "cac",

            # Behavior
            "session_duration", "bounce_rate",
            "pageviews", "unique_visitors",

            # Payment & shipping
            "payment_gateway", "shipping_method"
        },
        "ambiguous": {
            "user_id", "order_date",
            "discount_code", "amount"
        }
    },

    # =====================================================
    # CUSTOMER 🤝
    # =====================================================
    "customer": {
        "high": {
            # Identity
            "customer_id", "customer_name",

            # Segmentation
            "segment",

            # Value modeling
            "rfm", "recency", "frequency", "monetary",
            "lifetime_value",

            # Experience
            "churn", "nps", "csat",
            "support_ticket"
        },
        "ambiguous": {
            "email", "phone",
            "transaction_id", "amount", "date"
        }
    },

    # =====================================================
    # FINANCE 💰
    # =====================================================
    "finance": {
        "high": {
            # P&L
            "revenue", "expense", "profit", "loss",

            # Balance sheet
            "asset", "liability", "equity",

            # Cash & earnings
            "cash_flow", "net_income", "ebitda",

            # Market data
            "open", "close", "high", "low",
            "adjusted_close", "volume", "market_cap",

            # Rates
            "interest_rate", "balance"
        },
        "ambiguous": {
            "price", "amount", "currency", "date"
        }
    },

    # =====================================================
    # HEALTHCARE 🏥 (BOUNDARY-SAFE)
    # =====================================================
    "healthcare": {
        "high": {
            # Identity & encounters
            "patient_id", "encounter", "visit_id",

            # Dates
            "admission_date", "discharge_date", "fill_date",

            # Clinical
            "diagnosis", "treatment", "doctor", "bed_id",

            # Operations
            "length_of_stay", "los", "duration",

            # Outcomes
            "readmitted", "mortality", "flag",

            # Pharmacy
            "days_supply", "rx_volume", "supply",

            # Financial (healthcare-contextual)
            "billing_amount", "cost",

            # Population health
            "population"
        },
        "ambiguous": {
            "id", "date", "age", "gender",
            "facility", "status"
        }
    },
}
