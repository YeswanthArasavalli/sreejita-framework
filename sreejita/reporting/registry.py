# =====================================================
# REPORT ENGINE REGISTRY — v2.9 (EXTENDED)
# =====================================================

# -------------------------
# RETAIL
# -------------------------
from sreejita.reporting.retail.kpis import compute_retail_kpis
from sreejita.reporting.retail.insights import generate_retail_insights
from sreejita.reporting.retail.recommendations import generate_retail_recommendations
from sreejita.reporting.retail.narrative import get_domain_narrative as retail_narrative

# -------------------------
# CUSTOMER
# -------------------------
from sreejita.reporting.customer.kpis import compute_customer_kpis
from sreejita.reporting.customer.insights import generate_customer_insights
from sreejita.reporting.customer.recommendations import generate_customer_recommendations
from sreejita.reporting.customer.narrative import get_domain_narrative as customer_narrative

# -------------------------
# FINANCE
# -------------------------
from sreejita.reporting.finance.kpis import compute_finance_kpis
from sreejita.reporting.finance.insights import generate_finance_insights
from sreejita.reporting.finance.recommendations import generate_finance_recommendations
from sreejita.reporting.finance.narrative import get_domain_narrative as finance_narrative

# -------------------------
# OPS
# -------------------------
# OPS
from sreejita.reporting.ops.kpis import compute_ops_kpis
from sreejita.reporting.ops.insights import generate_ops_insights
from sreejita.reporting.ops.recommendations import generate_ops_recommendations
from sreejita.reporting.ops.narrative import get_domain_narrative as ops_narrative

# -------------------------
# HEALTHCARE
# -------------------------
# HEALTHCARE
from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis
from sreejita.reporting.healthcare.insights import generate_healthcare_insights
from sreejita.reporting.healthcare.recommendations import generate_healthcare_recommendations
from sreejita.reporting.healthcare.narrative import get_domain_narrative as healthcare_narrative

# -------------------------
# MARKETING
# -------------------------
# MARKETING
from sreejita.reporting.marketing.kpis import compute_marketing_kpis
from sreejita.reporting.marketing.insights import generate_marketing_insights
from sreejita.reporting.marketing.recommendations import generate_marketing_recommendations
from sreejita.reporting.marketing.narrative import get_domain_narrative as marketing_narrative


# =====================================================
# DOMAIN → ENGINE MAP
# =====================================================

DOMAIN_REPORT_ENGINES = {
    "retail": {
        "kpis": compute_retail_kpis,
        "insights": generate_retail_insights,
        "recommendations": generate_retail_recommendations,
    },
    "customer": {
        "kpis": compute_customer_kpis,
        "insights": generate_customer_insights,
        "recommendations": generate_customer_recommendations,
    },
    "finance": {
        "kpis": compute_finance_kpis,
        "insights": generate_finance_insights,
        "recommendations": generate_finance_recommendations,
    },
    "ops": {
        "kpis": compute_ops_kpis,
        "insights": generate_ops_insights,
        "recommendations": generate_ops_recommendations,
    },
    ''' "healthcare": {
        "kpis": compute_healthcare_kpis,
        "insights": generate_healthcare_insights,
        "recommendations": generate_healthcare_recommendations,
    },'''
    "marketing": {
    "kpis": compute_marketing_kpis,
    "insights": generate_marketing_insights,
    "recommendations": generate_marketing_recommendations,
    },

}

# =====================================================
# DOMAIN NARRATIVE REGISTRY
# =====================================================

DOMAIN_NARRATIVES = {
    "retail": retail_narrative,
    "customer": customer_narrative,
    "finance": finance_narrative,
    "healthcare": healthcare_narrative,
    "ops": ops_narrative,
    "marketing": marketing_narrative,
}

# =====================================================
# VISUAL REGISTRY (UNCHANGED)
# =====================================================
from sreejita.visuals.time_series import sales_trend_visual
from sreejita.visuals.categorical import category_sales_visual
from sreejita.visuals.correlation import shipping_vs_sales_visual

DOMAIN_VISUALS = {
    "retail": {
        "__always__": [
            sales_trend_visual,
            category_sales_visual,
            shipping_vs_sales_visual,
        ]
    }
}
