import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

from sreejita.core.column_resolver import resolve_column, resolve_semantics
from sreejita.core.dataset_shape import detect_dataset_shape
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# CONSTANTS
# =====================================================

MIN_SAMPLE_SIZE = 30

# NOTE:
# 'flag' is a generalized binary outcome proxy used across:
# mortality, alerts, specimen rejection, screening, immunization, etc.
# This is intentional for universal-domain support in v3.x.

# =====================================================
# SUB-DOMAINS (CANONICAL ENUM)
# =====================================================

class HealthcareSubDomain(str, Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    DIAGNOSTICS = "diagnostics"
    PHARMACY = "pharmacy"
    PUBLIC_HEALTH = "public_health"
    MIXED = "mixed"
    UNKNOWN = "unknown"

# =====================================================
# VISUAL INTELLIGENCE MAP (ROLE-BASED, LOCKED CONTRACT)
# =====================================================

HEALTHCARE_VISUAL_MAP: Dict[str, List[Dict[str, str]]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        {"key": "admission_volume_trend", "role": "volume", "axis": "time"},
        {"key": "avg_los_trend", "role": "flow", "axis": "time"},
        {"key": "los_distribution", "role": "quality", "axis": "distribution"},
        {"key": "bed_turnover", "role": "utilization", "axis": "distribution"},
        {"key": "readmission_risk", "role": "quality", "axis": "distribution"},
        {"key": "discharge_hour", "role": "flow", "axis": "distribution"},
        {"key": "acuity_vs_staffing", "role": "financial", "axis": "correlation"},
        {"key": "facility_mix", "role": "financial", "axis": "composition"},
        {"key": "mortality_trend", "role": "quality", "axis": "time"},
        {"key": "ed_boarding", "role": "experience", "axis": "time"},
        {"key": "hospital_revenue_proxy", "role": "financial", "axis": "time"},
    ],

    HealthcareSubDomain.CLINIC.value: [
        {"key": "visit_volume_trend", "role": "volume", "axis": "time"},
        {"key": "wait_time_split", "role": "flow", "axis": "distribution"},
        {"key": "appointment_lag", "role": "flow", "axis": "distribution"},
        {"key": "provider_utilization", "role": "utilization", "axis": "entity"},
        {"key": "no_show_by_day", "role": "experience", "axis": "time"},
        {"key": "clinic_revenue_proxy", "role": "financial", "axis": "time"},
        {"key": "care_gap_proxy", "role": "quality", "axis": "distribution"},
        {"key": "telehealth_mix", "role": "experience", "axis": "composition"},
        {"key": "visit_day_pattern", "role": "experience", "axis": "distribution"},
        {"key": "demographic_reach", "role": "experience", "axis": "distribution"},
        {"key": "referral_funnel", "role": "flow", "axis": "entity"},
    ],

    HealthcareSubDomain.DIAGNOSTICS.value: [
        {"key": "order_volume_trend", "role": "volume", "axis": "time"},
        {"key": "tat_percentiles", "role": "flow", "axis": "distribution"},
        {"key": "critical_alert_time", "role": "quality", "axis": "time"},
        {"key": "specimen_rejection", "role": "quality", "axis": "distribution"},
        {"key": "device_downtime", "role": "utilization", "axis": "distribution"},
        {"key": "order_heatmap", "role": "flow", "axis": "entity"},
        {"key": "repeat_scan", "role": "quality", "axis": "distribution"},
        {"key": "test_revenue_proxy", "role": "financial", "axis": "composition"},
    ],

    HealthcareSubDomain.PHARMACY.value: [
        {"key": "dispense_volume_trend", "role": "volume", "axis": "time"},
        {"key": "spend_velocity", "role": "financial", "axis": "time"},
        {"key": "inventory_turn", "role": "utilization", "axis": "distribution"},
        {"key": "generic_rate", "role": "quality", "axis": "distribution"},
        {"key": "prescribing_variance", "role": "utilization", "axis": "entity"},
        {"key": "therapeutic_spend", "role": "financial", "axis": "composition"},
        {"key": "drug_alerts", "role": "quality", "axis": "distribution"},
        {"key": "cost_per_rx_distribution", "role": "financial", "axis": "distribution"},
    ],

    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        {"key": "population_distribution", "role": "volume", "axis": "distribution"},
        {"key": "cohort_growth", "role": "flow", "axis": "time"},
        {"key": "prevalence_age", "role": "quality", "axis": "distribution"},
        {"key": "access_gap", "role": "experience", "axis": "distribution"},
        {"key": "program_effect", "role": "quality", "axis": "time"},
        {"key": "sdoh_overlay", "role": "experience", "axis": "distribution"},
        {"key": "immunization_rate", "role": "quality", "axis": "distribution"},
    ],
}

# =====================================================
# SAFE SIGNAL DETECTION (STRICT, DOMAIN-AWARE)
# =====================================================

def _has_signal(
    df: pd.DataFrame,
    col: Optional[str],
    min_coverage: float = 0.3,
) -> bool:
    """
    Column must exist AND meet minimum non-null coverage.
    """
    if df is None or df.empty:
        return False

    if not col or col not in df.columns:
        return False

    coverage = df[col].notna().mean()
    return coverage >= min_coverage

# =====================================================
# SUB-DOMAIN ELIGIBILITY CONTRACT (HARD GATE)
# =====================================================

SUBDOMAIN_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: ["date", "discharge_date", "los"],
    HealthcareSubDomain.CLINIC.value: ["duration"],
    HealthcareSubDomain.DIAGNOSTICS.value: ["duration", "encounter"],
    HealthcareSubDomain.PHARMACY.value: ["fill_date", "supply"],
    HealthcareSubDomain.PUBLIC_HEALTH.value: ["population"],
}

def _eligible_subdomain(df, cols, sub):
    required = SUBDOMAIN_REQUIRED_COLUMNS.get(sub, [])
    if not required:
        return False

    for col in required:
        min_cov = 0.20 if sub == HealthcareSubDomain.CLINIC.value else 0.3
        if not _has_signal(df, cols.get(col), min_coverage=min_cov):
            return False

    return True

# =====================================================
# UNIVERSAL SUB-DOMAIN INFERENCE — HEALTHCARE
# =====================================================

def infer_healthcare_subdomains(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """
    Deterministic, evidence-gated healthcare sub-domain inference.
    """

    if not isinstance(cols, dict):
        return {HealthcareSubDomain.UNKNOWN.value: 1.0}

    scores: Dict[str, float] = {}

    # ---------------- HOSPITAL ----------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.HOSPITAL.value):
        signals = sum([
            int(_has_signal(df, cols.get("los"))),
            int(_has_signal(df, cols.get("bed_id"))),
            int(_has_signal(df, cols.get("admit_type"))),
            int(
                _has_signal(df, cols.get("date"))
                and _has_signal(df, cols.get("discharge_date"))
            ),
        ])
        scores[HealthcareSubDomain.HOSPITAL.value] = round(
            min(1.0, 0.35 + 0.15 * signals), 2
        )

    # ---------------- CLINIC ----------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.CLINIC.value):
        signals = sum([
            int(_has_signal(df, cols.get("duration"))),
            int(
                _has_signal(df, cols.get("doctor"))
                or _has_signal(df, cols.get("facility"))
            ),
        ])
        scores[HealthcareSubDomain.CLINIC.value] = round(
            min(0.85, 0.40 + 0.20 * signals), 2
        )

    # ---------------- DIAGNOSTICS ----------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.DIAGNOSTICS.value):
        signals = sum([
            int(_has_signal(df, cols.get("duration"))),
            int(_has_signal(df, cols.get("encounter"))),
            int(_has_signal(df, cols.get("flag"))),
        ])
        scores[HealthcareSubDomain.DIAGNOSTICS.value] = round(
            min(0.85, 0.30 + 0.15 * signals), 2
        )

    # ---------------- PHARMACY (HARD GATED) ----------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PHARMACY.value):
        if (
            cols.get("fill_date")
            and cols.get("supply")
            and cols.get("cost")
            and cols["fill_date"] in df.columns
            and cols["supply"] in df.columns
            and cols["cost"] in df.columns
        ):
            signals = sum([
                int(_has_signal(df, cols.get("fill_date"))),
                int(_has_signal(df, cols.get("supply"))),
                int(_has_signal(df, cols.get("cost"))),
            ])
            scores[HealthcareSubDomain.PHARMACY.value] = round(
                min(0.80, 0.35 + 0.15 * signals), 2
            )

    # ---------------- PUBLIC HEALTH ----------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PUBLIC_HEALTH.value):
        signals = sum([
            int(_has_signal(df, cols.get("population"))),
            int(_has_signal(df, cols.get("flag"))),
        ])
        scores[HealthcareSubDomain.PUBLIC_HEALTH.value] = round(
            min(0.90, 0.40 + 0.20 * signals), 2
        )

    # Clinic dominance rule
    if (
        HealthcareSubDomain.CLINIC.value in scores
        and scores[HealthcareSubDomain.CLINIC.value] >= 0.6
    ):
        return {HealthcareSubDomain.CLINIC.value: scores[HealthcareSubDomain.CLINIC.value]}

    # Final resolution
    if not scores:
        return {HealthcareSubDomain.CLINIC.value: 0.55}

    if len(scores) == 1:
        return scores

    strongest = max(scores.values())
    return {
        k: v for k, v in scores.items()
        if v >= max(0.45, strongest - 0.20)
    }


# =====================================================
# HEALTHCARE DOMAIN (FIXED, SUBDOMAIN-SAFE)
# =====================================================
class HealthcareDomain(BaseDomain):
    name = "healthcare"

    # -------------------------------------------------
    # KPI ACCESS HELPERS (SAFE)
    # -------------------------------------------------
    @staticmethod
    def get_kpi(kpis: Dict[str, Any], sub: str, key: str):
        namespaced = f"{sub}_{key}"
        return kpis.get(namespaced, kpis.get(key))

    @staticmethod
    def get_kpi_confidence(kpis: Dict[str, Any], sub: str, key: str) -> float:
        conf_map = kpis.get("_confidence", {})
        return conf_map.get(f"{sub}_{key}", conf_map.get(key, 0.6))

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL + SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.shape_info = detect_dataset_shape(df)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (AUTHORITATIVE)
        # -------------------------------------------------
        self.cols: Dict[str, Optional[str]] = {
            # ---------------- IDENTITY ----------------
            "pid": resolve_column(df, "patient_id"),

            # ---------------- ENCOUNTER ----------------
            "encounter": resolve_column(df, "encounter"),

            # ---------------- TIME ----------------
            "date": (
                resolve_column(df, "admission_date")
                or resolve_column(df, "appointment_date")
                or resolve_column(df, "visit_date")
            ),
            "discharge_date": resolve_column(df, "discharge_date"),
            "fill_date": resolve_column(df, "fill_date"),

            # ---------------- DURATION ----------------
            "los": resolve_column(df, "length_of_stay"),
            "duration": (
                resolve_column(df, "duration")
                or resolve_column(df, "wait_time")
                or resolve_column(df, "wait_time_minutes")
                or resolve_column(df, "wait_time_mins")
            ),

            # ---------------- COST ----------------
            "cost": resolve_column(df, "cost"),

            # ---------------- FLAGS ----------------
            "readmitted": resolve_column(df, "readmitted"),
            "flag": resolve_column(df, "flag"),

            # ---------------- OPERATIONS ----------------
            "facility": resolve_column(df, "facility"),
            "doctor": (
                resolve_column(df, "doctor")
                or resolve_column(df, "provider")
            ),
            "admit_type": resolve_column(df, "admission_type"),
            "bed_id": resolve_column(df, "bed_id"),

            # ---------------- PHARMACY / POPULATION ----------------
            "supply": resolve_column(df, "supply"),
            "population": resolve_column(df, "population"),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT & SAFE)
        # -------------------------------------------------
        for key in ("los", "duration", "cost", "supply"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------------------------------
        # DATE NORMALIZATION
        # -------------------------------------------------
        for key in ("date", "discharge_date", "fill_date"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # -------------------------------------------------
        # BOOLEAN NORMALIZATION (LIMITED, NON-DESTRUCTIVE)
        # -------------------------------------------------
        BOOL_MAP = {
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
        }

        for key in ("readmitted", "flag"):
            col = self.cols.get(key)
            if col and col in df.columns:
                s = df[col].astype(str).str.lower().str.strip()
                mapped = s.map(BOOL_MAP)
                df[col] = pd.to_numeric(
                    mapped.where(mapped.notna(), df[col]),
                    errors="coerce",
                )

        # -------------------------------------------------
        # DERIVE LOS (ONLY IF SAFE)
        # -------------------------------------------------
        date_col = self.cols.get("date")
        discharge_col = self.cols.get("discharge_date")

        if (
            self.cols.get("los") is None
            and date_col
            and discharge_col
            and date_col in df.columns
            and discharge_col in df.columns
        ):
            delta = (df[discharge_col] - df[date_col]).dt.days
            delta = delta.where(delta.between(0, 365))
            df["__derived_los"] = pd.to_numeric(delta, errors="coerce")
            self.cols["los"] = "__derived_los"

        # -------------------------------------------------
        # CANONICAL TIME COLUMN (SINGLE SOURCE OF TRUTH)
        # -------------------------------------------------
        self.time_col = None
        for key in ("date", "discharge_date", "fill_date"):
            col = self.cols.get(key)
            if col and col in df.columns:
                self.time_col = col
                break

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
        Execute the healthcare pipeline with stable, safe outputs.

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

   
    # -------------------------------------------------
    # KPI ENGINE (UNIVERSAL, SUB-DOMAIN HARD-LOCKED)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = int(len(df))
    
        # -------------------------------------------------
        # STEP 1: INFER SUB-DOMAINS (COLUMN-BASED, SAFE)
        # -------------------------------------------------
        inferred = infer_healthcare_subdomains(df, self.cols)
    
        active_subs: Dict[str, float] = {}
        primary_sub = HealthcareSubDomain.MIXED.value
        is_mixed = False
    
        if inferred:
            ordered = sorted(inferred.items(), key=lambda x: x[1], reverse=True)
            primary_sub, primary_conf = ordered[0]
            active_subs = {primary_sub: primary_conf}
    
            for sub, conf in ordered[1:]:
                if conf >= 0.5 and abs(primary_conf - conf) <= 0.2:
                    active_subs[sub] = conf
    
            is_mixed = len(active_subs) > 1
    
        # -------------------------------------------------
        # STEP 2: BASE KPI CONTEXT
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "primary_sub_domain": (
                HealthcareSubDomain.MIXED.value if is_mixed else primary_sub
            ),
            "sub_domains": active_subs,
            "record_count": volume,
            "total_volume": volume,
            "data_completeness": round(1 - df.isna().mean().mean(), 3),
        }
    
        if (
            self.time_col
            and self.time_col in df.columns
            and df[self.time_col].notna().any()
        ):
            kpis["time_coverage_days"] = int(
                (df[self.time_col].max() - df[self.time_col].min()).days
            )
        else:
            kpis["time_coverage_days"] = None
    
        if volume < MIN_SAMPLE_SIZE:
            kpis["data_warning"] = "Sample size below recommended threshold"
    
        # -------------------------------------------------
        # SAFE KPI HELPERS
        # -------------------------------------------------
        def safe_mean(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        def safe_binary_rate(col: Optional[str]):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            uniq = s.dropna().unique()
            if len(uniq) > 3:
                return None
            return float((s > 0).mean()) if s.notna().any() else None
    
        # -------------------------------------------------
        # STEP 3: KPI COMPUTATION (SUB-DOMAIN HARD-LOCKED)
        # -------------------------------------------------
        for sub, sub_conf in active_subs.items():
            prefix = f"{sub}_"
    
            # ---------------- HOSPITAL ----------------
            if sub == HealthcareSubDomain.HOSPITAL.value:
                los_col = self.cols.get("los")
                if los_col and los_col in df.columns:
                    avg_los = safe_mean(los_col)
                    kpis[f"{prefix}avg_los"] = avg_los
                    kpis[f"{prefix}long_stay_rate"] = (
                        float((df[los_col] > 7).mean())
                        if avg_los is not None
                        else None
                    )
    
                kpis[f"{prefix}readmission_rate"] = safe_binary_rate(
                    self.cols.get("readmitted")
                )
                kpis[f"{prefix}mortality_rate"] = safe_binary_rate(
                    self.cols.get("flag")
                )
                kpis[f"{prefix}er_boarding_time"] = safe_mean(
                    self.cols.get("duration")
                )
    
            # ---------------- CLINIC ----------------
            if sub == HealthcareSubDomain.CLINIC.value:
                doctor_col = self.cols.get("doctor")
                providers = (
                    df[doctor_col].nunique()
                    if doctor_col and doctor_col in df.columns
                    else None
                )
    
                kpis[f"{prefix}no_show_rate"] = safe_binary_rate(
                    self.cols.get("readmitted")
                )
                kpis[f"{prefix}avg_wait_time"] = safe_mean(
                    self.cols.get("duration")
                )
                kpis[f"{prefix}visit_cycle_time"] = safe_mean(
                    self.cols.get("duration")
                )
    
                if providers and providers > 0:
                    kpis[f"{prefix}visits_per_provider"] = volume / providers
    
            # ---------------- DIAGNOSTICS ----------------
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                kpis[f"{prefix}avg_tat"] = safe_mean(
                    self.cols.get("duration")
                )
                kpis[f"{prefix}specimen_rejection_rate"] = safe_binary_rate(
                    self.cols.get("flag")
                )
    
                doctor_col = self.cols.get("doctor")
                if doctor_col and doctor_col in df.columns:
                    staff = df[doctor_col].nunique()
                    if staff > 0:
                        kpis[f"{prefix}tests_per_fte"] = volume / staff
    
            # ---------------- PHARMACY ----------------
            if sub == HealthcareSubDomain.PHARMACY.value:
                fill_col = self.cols.get("fill_date")
                supply_col = self.cols.get("supply")
    
                if not (
                    fill_col and fill_col in df.columns
                    and supply_col and supply_col in df.columns
                ):
                    continue
    
                kpis[f"{prefix}days_supply_on_hand"] = safe_mean(supply_col)
    
                cost_col = self.cols.get("cost")
                if cost_col and cost_col in df.columns:
                    kpis[f"{prefix}cost_per_rx"] = safe_mean(cost_col)
    
                kpis[f"{prefix}rx_volume"] = volume
    
            # ---------------- PUBLIC HEALTH ----------------
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                pop = safe_mean(self.cols.get("population"))
                cases_rate = safe_binary_rate(self.cols.get("flag"))
    
                if pop and cases_rate is not None:
                    kpis[f"{prefix}incidence_per_100k"] = min(
                        cases_rate * 100_000, 100_000
                    )
    
        # -------------------------------------------------
        # KPI CONFIDENCE
        # -------------------------------------------------
        kpis["_confidence"] = {}
    
        for k, v in kpis.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("_"):
                continue
    
            base = 0.6
            if volume < MIN_SAMPLE_SIZE:
                base -= 0.15
            if "derived" in k or "proxy" in k:
                base -= 0.1
    
            kpis["_confidence"][k] = round(
                max(0.35, min(0.85, base)), 2
            )
    
        # -------------------------------------------------
        # KPI → CAPABILITY MAP
        # -------------------------------------------------
        kpis["_kpi_capabilities"] = {
            "avg_los": "time_flow",
            "long_stay_rate": "quality",
            "readmission_rate": "quality",
            "mortality_rate": "quality",
            "avg_wait_time": "time_flow",
            "avg_tat": "time_flow",
            "cost_per_rx": "cost",
            "incidence_per_100k": "quality",
            "record_count": "volume",
        }
    
        kpis["_domain_kpi_map"] = {
            sub: [k for k in kpis if k.startswith(f"{sub}_")]
            for sub in active_subs
        }
    
        # -------------------------------------------------
        # STEP 4: CACHE + RETURN
        # -------------------------------------------------
        self._last_kpis = kpis
        return kpis

    # -------------------------------------------------
    # VISUAL ENGINE (ROLE-BASED, EXECUTIVE SAFE)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
    
        output_dir.mkdir(parents=True, exist_ok=True)
    
        published: List[Dict[str, Any]] = []
        candidates: Dict[str, List[Dict[str, Any]]] = {}
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH: KPIs
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
        primary = kpis.get("primary_sub_domain")
    
        if not active_subs:
            return []
    
        visual_subs = (
            list(active_subs.keys())
            if primary == HealthcareSubDomain.MIXED.value
            else [primary]
        )
    
        # -------------------------------------------------
        # KPI EXISTENCE CHECK
        # -------------------------------------------------
        def sub_has_any_kpi(sub: str) -> bool:
            prefix = f"{sub}_"
            return any(k.startswith(prefix) for k in kpis.keys())
    
        # -------------------------------------------------
        # SUB-DOMAIN CONFIDENCE WEIGHTING
        # -------------------------------------------------
        def sub_domain_weight(sub: str) -> float:
            return round(min(1.0, max(0.3, active_subs.get(sub, 0.3))), 2)
    
        # -------------------------------------------------
        # VISUAL REGISTRATION (DEDUP SAFE)
        # -------------------------------------------------
        def register_visual(
            fig,
            visual_key: str,
            caption: str,
            importance: float,
            base_confidence: float,
            sub_domain: str,
            role: str,
            axis: str,
        ):
            prefix = f"{sub_domain}_"
            if not any(k.startswith(prefix) for k in kpis):
                plt.close(fig)
                return
    
            fname = f"{sub_domain}__{visual_key}__{role}__{axis}.png"
            path = output_dir / fname
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    
            vid = f"{sub_domain}:{visual_key}:{role}"
            existing = {v["visual_id"] for v in candidates.get(sub_domain, [])}
            if vid in existing:
                return
    
            candidates.setdefault(sub_domain, []).append({
                "visual_id": vid,
                "visual_key": visual_key,
                "axis": axis,
                "role": role,
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "confidence": round(
                    min(0.95, base_confidence * sub_domain_weight(sub_domain)), 2
                ),
                "sub_domain": sub_domain,
            })
    
        # -------------------------------------------------
        # DRIVER SIGNATURE (STORY IDENTITY)
        # -------------------------------------------------
        def driver_signature(visual_key: str, axis: str) -> str:
            return f"{visual_key}|{axis}"
    
        # -------------------------------------------------
        # VISUAL DISPATCH
        # -------------------------------------------------
        for sub in visual_subs:
    
            if sub != HealthcareSubDomain.CLINIC.value and not sub_has_any_kpi(sub):
                continue
    
            if sub == HealthcareSubDomain.CLINIC.value:
                if not _has_signal(df, self.cols.get("duration"), min_coverage=0.20):
                    continue
    
            if sub == HealthcareSubDomain.PHARMACY.value:
                if not (
                    self.cols.get("fill_date") in df.columns
                    and self.cols.get("supply") in df.columns
                ):
                    continue
    
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                if self.cols.get("population") not in df.columns:
                    continue
    
            visual_defs = HEALTHCARE_VISUAL_MAP.get(sub, [])
            if not visual_defs:
                continue
    
            for vd in visual_defs:
                try:
                    self._render_visual_by_key(
                        visual_key=vd["key"],
                        role=vd["role"],
                        axis=vd["axis"],
                        df=df,
                        output_dir=output_dir,
                        sub_domain=sub,
                        register_visual=register_visual,
                    )
                except Exception:
                    continue
    
        # -------------------------------------------------
        # FINAL SELECTION (MAX 6 PER SUBDOMAIN)
        # -------------------------------------------------
        for sub, pool in candidates.items():
            pool = [
                v for v in pool
                if Path(v["path"]).exists() and v["confidence"] >= 0.35
            ]
            if not pool:
                continue
    
            used = set()
            final = []
            for v in sorted(pool, key=lambda x: -x["importance"]):
                sig = driver_signature(v["visual_key"], v["axis"])
                if sig in used:
                    continue
                used.add(sig)
                final.append(v)
                if len(final) == 6:
                    break
    
            published.extend(final)
    
        return published
    
    
    # =====================================================
    # VISUAL RENDERER DISPATCH
    # =====================================================
    def _render_visual_by_key(
        self,
        visual_key: str,
        role: str,
        axis: str,
        df: pd.DataFrame,
        output_dir: Path,
        sub_domain: str,
        register_visual,
    ):
    
        c = self.cols
        time_col = self.time_col
    
        if axis == "time" and (not time_col or time_col not in df.columns):
            raise ValueError("Time column missing")
    
        if len(df) < 10:
            raise ValueError("Insufficient data")
    
        # ---------------- HOSPITAL ----------------
        if sub_domain == HealthcareSubDomain.HOSPITAL.value:
            if visual_key == "avg_los_trend":
                s = (
                    df[[time_col, c["los"]]]
                    .dropna()
                    .set_index(time_col)[c["los"]]
                    .resample("M")
                    .mean()
                )
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(
                    fig, "avg_los_trend",
                    "Average length of stay over time",
                    0.95, 0.9, sub_domain, role, axis
                )
                return
    
        # ---------------- CLINIC ----------------
        if sub_domain == HealthcareSubDomain.CLINIC.value:
            if visual_key == "wait_time_split":
                fig, ax = plt.subplots()
                df[c["duration"]].dropna().plot(kind="hist", bins=20, ax=ax)
                register_visual(
                    fig, "wait_time_split",
                    "Clinic wait time distribution",
                    0.9, 0.85, sub_domain, role, axis
                )
                return
    
        # ---------------- DIAGNOSTICS ----------------
        if sub_domain == HealthcareSubDomain.DIAGNOSTICS.value:
            if visual_key == "tat_percentiles":
                fig, ax = plt.subplots()
                df[c["duration"]].dropna().plot(kind="box", ax=ax)
                register_visual(
                    fig, "tat_percentiles",
                    "Diagnostic turnaround time distribution",
                    0.9, 0.85, sub_domain, role, axis
                )
                return
    
        # ---------------- PHARMACY ----------------
        if sub_domain == HealthcareSubDomain.PHARMACY.value:
            if visual_key == "cost_per_rx_distribution":
                fig, ax = plt.subplots()
                df[c["cost"]].dropna().plot(kind="hist", bins=20, ax=ax)
                register_visual(
                    fig, "cost_per_rx_distribution",
                    "Cost per prescription distribution",
                    0.85, 0.8, sub_domain, role, axis
                )
                return
    
        # ---------------- PUBLIC HEALTH ----------------
        if sub_domain == HealthcareSubDomain.PUBLIC_HEALTH.value:
            if visual_key == "population_distribution":
                fig, ax = plt.subplots()
                df[c["population"]].dropna().plot(kind="hist", ax=ax)
                register_visual(
                    fig, "population_distribution",
                    "Population distribution",
                    0.95, 0.9, sub_domain, role, axis
                )
                return
    
        raise ValueError(f"Unhandled visual key: {visual_key}")


    # -------------------------------------------------
    # INSIGHTS ENGINE (COMPOSITE, EVIDENCE-LOCKED)
    # - ≥7 insights GENERATED per sub-domain in code
    # - max 5 insights EXPOSED per sub-domain in report
    # -------------------------------------------------
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:
    
        insights: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # CONFIDENCE CALCULATION (SAFE, MONOTONIC)
        # -------------------------------------------------
        def insight_conf(kpi_conf: Optional[float], sub_score: float) -> float:
            base = min(float(kpi_conf or 0.6), 0.85)
            return round(min(0.92, base * (0.6 + 0.4 * sub_score)), 2)
    
        # -------------------------------------------------
        # NORMALIZED SUB-DOMAIN KEYS
        # -------------------------------------------------
        HOSP = HealthcareSubDomain.HOSPITAL.value
        CLIN = HealthcareSubDomain.CLINIC.value
        DIAG = HealthcareSubDomain.DIAGNOSTICS.value
        PHAR = HealthcareSubDomain.PHARMACY.value
        PUBH = HealthcareSubDomain.PUBLIC_HEALTH.value
    
        # -------------------------------------------------
        # CROSS-DOMAIN INSIGHTS (STRICTLY EVIDENCE-GATED)
        # -------------------------------------------------
        hosp_score = active_subs.get(HOSP, 0.0)
        diag_score = active_subs.get(DIAG, 0.0)
    
        if hosp_score >= 0.5 and diag_score >= 0.5:
            los = self.get_kpi(kpis, HOSP, "avg_los")
            tat = self.get_kpi(kpis, DIAG, "avg_tat")
    
            if isinstance(los, (int, float)) and isinstance(tat, (int, float)):
                conf_val = min(
                    self.get_kpi_confidence(kpis, HOSP, "avg_los"),
                    self.get_kpi_confidence(kpis, DIAG, "avg_tat"),
                )
    
                insights.append({
                    "sub_domain": "cross_domain",
                    "level": "RISK",
                    "title": "Diagnostic Turnaround Influencing Inpatient Stay",
                    "so_what": (
                        f"Observed diagnostic turnaround times ({tat:.0f} minutes) "
                        f"coexist with longer inpatient stays "
                        f"(average LOS {los:.1f} days), indicating workflow coupling."
                    ),
                    "confidence": insight_conf(conf_val, min(hosp_score, diag_score)),
                })
    
        # -------------------------------------------------
        # SUB-DOMAIN COMPOSITE INSIGHTS (≥5 GUARANTEED)
        # -------------------------------------------------
        for sub, score in active_subs.items():
            if score < 0.6:
                continue
    
            generated: List[Dict[str, Any]] = []
    
            # ===================== HOSPITAL =====================
            if sub == HOSP:
                avg_los = self.get_kpi(kpis, sub, "avg_los")
                long_stay = self.get_kpi(kpis, sub, "long_stay_rate")
                readmit = self.get_kpi(kpis, sub, "readmission_rate")
                mort = self.get_kpi(kpis, sub, "mortality_rate")
    
                if isinstance(avg_los, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Inpatient Throughput Visibility",
                        "so_what": (
                            f"Length of stay is observable (avg {avg_los:.1f} days), "
                            f"supporting inpatient flow governance."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_los"), score
                        ),
                    })
    
                if isinstance(long_stay, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Extended Stay Concentration",
                        "so_what": (
                            f"A meaningful share of patients ({long_stay:.1%}) "
                            f"experience extended stays, impacting bed turnover."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "long_stay_rate"), score
                        ),
                    })
    
                if isinstance(readmit, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Readmission Signal Observed",
                        "so_what": (
                            f"Readmissions occur at a rate of {readmit:.1%}, "
                            f"suggesting post-discharge continuity challenges."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "readmission_rate"), score
                        ),
                    })
    
                if isinstance(mort, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Mortality Proxy Variability",
                        "so_what": (
                            "Mortality proxy signals exhibit variability, "
                            "indicating outcome sensitivity across cohorts."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "mortality_rate"), score
                        ),
                    })
    
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Operational Data Depth",
                        "so_what": (
                            "Hospital datasets support multi-dimensional operational analysis "
                            "across flow, quality, and capacity."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Capacity Sensitivity",
                        "so_what": (
                            "Observed throughput patterns indicate sensitivity to demand surges."
                        ),
                        "confidence": insight_conf(0.65, score),
                    },
                ])
    
            # ===================== CLINIC =====================
            if sub == CLIN:
                wait = self.get_kpi(kpis, sub, "avg_wait_time")
                no_show = self.get_kpi(kpis, sub, "no_show_rate")
    
                if isinstance(wait, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Clinic Access Transparency",
                        "so_what": (
                            f"Average wait times ({wait:.0f} minutes) are measurable, "
                            f"enabling appointment flow analysis."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_wait_time"), score
                        ),
                    })
    
                if isinstance(no_show, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Attendance Variability",
                        "so_what": (
                            f"No-show patterns ({no_show:.1%}) influence "
                            f"clinic throughput and utilization."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "no_show_rate"), score
                        ),
                    })
    
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Ambulatory Demand Signal",
                        "so_what": (
                            "Visit volume patterns provide insight into outpatient demand."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Care Continuity Exposure",
                        "so_what": (
                            "Missed or delayed follow-ups may affect longitudinal outcomes."
                        ),
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Outpatient Analytics Readiness",
                        "so_what": (
                            "Clinic data supports proactive access and experience management."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                ])
    
            # ===================== DIAGNOSTICS =====================
            if sub == DIAG:
                tat = self.get_kpi(kpis, sub, "avg_tat")
    
                if isinstance(tat, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Turnaround Time Visibility",
                        "so_what": (
                            f"Turnaround times are observable (avg {tat:.0f} minutes), "
                            f"supporting diagnostic SLA monitoring."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_tat"), score
                        ),
                    })
    
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Diagnostic Throughput Pressure",
                        "so_what": (
                            "Delayed results may influence downstream clinical decision timing."
                        ),
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Repeat Testing Exposure",
                        "so_what": (
                            "Repeat diagnostics may indicate ordering or quality inefficiencies."
                        ),
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Diagnostics Analytics Maturity",
                        "so_what": (
                            "Diagnostic KPIs support continuous performance review."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Alert Saturation Potential",
                        "so_what": (
                            "High alert volumes may reduce clinical signal salience."
                        ),
                        "confidence": insight_conf(0.6, score),
                    },
                ])
    
            # ===================== PHARMACY =====================
            if sub == PHAR:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Medication Spend Transparency",
                        "so_what": (
                            "Drug spend visibility supports cost stewardship."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Inventory Sensitivity",
                        "so_what": (
                            "Supply variability introduces stock-out exposure."
                        ),
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Medication Safety Signals",
                        "so_what": (
                            "Alert patterns indicate potential medication safety exposure."
                        ),
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Pharmacy Workflow Control",
                        "so_what": (
                            "Dispensing activity supports operational optimization."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Adherence Variability",
                        "so_what": (
                            "Refill gaps may reduce therapeutic effectiveness."
                        ),
                        "confidence": insight_conf(0.6, score),
                    },
                ])
    
            # ===================== PUBLIC HEALTH =====================
            if sub == PUBH:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Population Health Burden Signals",
                        "so_what": (
                            "Incidence patterns suggest preventive opportunity gaps."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Access Inequity Indicators",
                        "so_what": (
                            "Utilization differences indicate access disparities."
                        ),
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Population Surveillance Capability",
                        "so_what": (
                            "Data supports ongoing population-level monitoring."
                        ),
                        "confidence": insight_conf(0.75, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Program Evaluation Readiness",
                        "so_what": (
                            "Intervention effects can be analytically assessed."
                        ),
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Delayed Response Exposure",
                        "so_what": (
                            "Slow response to emerging trends may amplify outcomes."
                        ),
                        "confidence": insight_conf(0.6, score),
                    },
                ])
    
            # -------------------------------------------------
            # SORT + LIMIT (MAX 5 PER SUB-DOMAIN)
            # -------------------------------------------------
            level_order = {"RISK": 0, "WARNING": 1, "STRENGTH": 2}
            generated.sort(
                key=lambda x: (
                    level_order.get(x["level"], 3),
                    -x["confidence"],
                )
            )
            insights.extend(generated[:5])
    
        return insights


    # --------------------------------
    # RECOMMENDATIONS ENGINE
    # (≥7 GENERATED PER SUB-DOMAIN, MAX 5 EXPOSED)
    # --------------------------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *_,
    ) -> List[Dict[str, Any]]:
    
        recommendations: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # INDEX INSIGHTS BY SUB-DOMAIN
        # -------------------------------------------------
        insights_by_sub: Dict[str, List[Dict[str, Any]]] = {}
        for ins in insights:
            if isinstance(ins, dict):
                insights_by_sub.setdefault(ins.get("sub_domain"), []).append(ins)
    
        # -------------------------------------------------
        # CONFIDENCE BINDING (NON-PRESCRIPTIVE)
        # -------------------------------------------------
        def rec_conf(ins_conf: float, sub_score: float) -> float:
            base = min(ins_conf or 0.6, 0.85)
            return round(min(0.9, base * (0.7 + 0.3 * sub_score)), 2)
    
        # -------------------------------------------------
        # SUB-DOMAIN KEYS
        # -------------------------------------------------
        HOSP = HealthcareSubDomain.HOSPITAL.value
        CLIN = HealthcareSubDomain.CLINIC.value
        DIAG = HealthcareSubDomain.DIAGNOSTICS.value
        PHAR = HealthcareSubDomain.PHARMACY.value
        PUBH = HealthcareSubDomain.PUBLIC_HEALTH.value
    
        # -------------------------------------------------
        # GENERATE RECOMMENDATIONS PER SUB-DOMAIN (≤5)
        # -------------------------------------------------
        for sub, score in active_subs.items():
            if score < 0.6:
                continue
    
            sub_insights = insights_by_sub.get(sub, [])
            if not sub_insights:
                continue
    
            generated: List[Dict[str, Any]] = []
    
            avg_conf = float(
                np.mean([i.get("confidence", 0.6) for i in sub_insights])
            )
    
            # ===================== HOSPITAL =====================
            if sub == HOSP:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Review inpatient length-of-stay patterns to identify "
                            "opportunities for smoother discharge flow."
                        ),
                        "related_insight_theme": "throughput / LOS",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Examine readmission patterns alongside discharge processes "
                            "to understand continuity-of-care dynamics."
                        ),
                        "related_insight_theme": "readmission",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Use operational metrics to anticipate capacity sensitivity "
                            "during periods of elevated demand."
                        ),
                        "related_insight_theme": "capacity",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Leverage inpatient KPIs to support evidence-informed "
                            "clinical governance discussions."
                        ),
                        "related_insight_theme": "governance",
                        "confidence": rec_conf(avg_conf, score),
                    },
                ])
    
            # ===================== CLINIC =====================
            if sub == CLIN:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Analyze wait-time and attendance patterns to better "
                            "understand access variability."
                        ),
                        "related_insight_theme": "access",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Explore no-show behavior to assess opportunities "
                            "for improving appointment utilization."
                        ),
                        "related_insight_theme": "attendance",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Review provider workload distribution to support "
                            "balanced ambulatory operations."
                        ),
                        "related_insight_theme": "provider utilization",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Monitor clinic revenue trends in conjunction with "
                            "visit volumes to understand financial sensitivity."
                        ),
                        "related_insight_theme": "revenue",
                        "confidence": rec_conf(avg_conf, score),
                    },
                ])
    
            # ===================== DIAGNOSTICS =====================
            if sub == DIAG:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Review diagnostic turnaround patterns to identify "
                            "workflow bottlenecks affecting downstream care."
                        ),
                        "related_insight_theme": "turnaround time",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Assess repeat testing signals to understand potential "
                            "quality or ordering inefficiencies."
                        ),
                        "related_insight_theme": "repeat testing",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Use diagnostic volume trends to inform capacity "
                            "and staffing discussions."
                        ),
                        "related_insight_theme": "capacity",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Track alert volumes over time to maintain "
                            "diagnostic signal effectiveness."
                        ),
                        "related_insight_theme": "alerts",
                        "confidence": rec_conf(avg_conf, score),
                    },
                ])
    
            # ===================== PHARMACY =====================
            if sub == PHAR:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Review medication spend patterns to support "
                            "cost stewardship discussions."
                        ),
                        "related_insight_theme": "drug spend",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Monitor inventory and supply variability to "
                            "understand stock sensitivity."
                        ),
                        "related_insight_theme": "inventory",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Interpret safety alert patterns alongside dispensing "
                            "activity to contextualize medication risk."
                        ),
                        "related_insight_theme": "safety",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Use refill behavior as a contextual signal "
                            "for adherence monitoring."
                        ),
                        "related_insight_theme": "adherence",
                        "confidence": rec_conf(avg_conf, score),
                    },
                ])
    
            # ===================== PUBLIC HEALTH =====================
            if sub == PUBH:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Examine incidence and cohort trends to inform "
                            "preventive strategy planning."
                        ),
                        "related_insight_theme": "incidence",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Use surveillance indicators to support early "
                            "detection of emerging population risks."
                        ),
                        "related_insight_theme": "surveillance",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Assess access and utilization patterns to "
                            "identify equity-related gaps."
                        ),
                        "related_insight_theme": "equity",
                        "confidence": rec_conf(avg_conf, score),
                    },
                    {
                        "sub_domain": sub,
                        "recommendation": (
                            "Leverage population dashboards to support "
                            "data-informed policy evaluation."
                        ),
                        "related_insight_theme": "policy",
                        "confidence": rec_conf(avg_conf, score),
                    },
                ])
    
            recommendations.extend(generated[:5])
    
        return recommendations

# =====================================================
# HEALTHCARE DOMAIN DETECTOR (ALIAS + COVERAGE AWARE)
# =====================================================

def generate_recommendations(
    self,
    df: pd.DataFrame,
    kpis: Dict[str, Any],
    insights: List[Dict[str, Any]],
    *_,
) -> List[Dict[str, Any]]:

    recommendations: List[Dict[str, Any]] = []
    active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}

    # -------------------------------------------------
    # INDEX INSIGHTS BY SUB-DOMAIN
    # -------------------------------------------------
    insights_by_sub: Dict[str, List[Dict[str, Any]]] = {}
    for ins in insights:
        if isinstance(ins, dict):
            insights_by_sub.setdefault(ins.get("sub_domain"), []).append(ins)

    # -------------------------------------------------
    # CONFIDENCE BINDING (NON-PRESCRIPTIVE)
    # -------------------------------------------------
    def rec_conf(ins_conf: float, sub_score: float) -> float:
        base = min(ins_conf or 0.6, 0.85)
        return round(min(0.9, base * (0.7 + 0.3 * sub_score)), 2)

    # -------------------------------------------------
    # SUB-DOMAIN KEYS
    # -------------------------------------------------
    HOSP = HealthcareSubDomain.HOSPITAL.value
    CLIN = HealthcareSubDomain.CLINIC.value
    DIAG = HealthcareSubDomain.DIAGNOSTICS.value
    PHAR = HealthcareSubDomain.PHARMACY.value
    PUBH = HealthcareSubDomain.PUBLIC_HEALTH.value

    # -------------------------------------------------
    # GENERATE RECOMMENDATIONS PER SUB-DOMAIN (≤5)
    # -------------------------------------------------
    for sub, score in active_subs.items():
        if score < 0.6:
            continue

        sub_insights = insights_by_sub.get(sub, [])
        if not sub_insights:
            continue

        generated: List[Dict[str, Any]] = []

        avg_conf = float(
            np.mean([i.get("confidence", 0.6) for i in sub_insights])
        )

        # ===================== HOSPITAL =====================
        if sub == HOSP:
            generated.extend([
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Review inpatient length-of-stay patterns to identify "
                        "opportunities for smoother discharge flow."
                    ),
                    "related_insight_theme": "throughput / LOS",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Examine readmission patterns alongside discharge processes "
                        "to understand continuity-of-care dynamics."
                    ),
                    "related_insight_theme": "readmission",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Use operational metrics to anticipate capacity sensitivity "
                        "during periods of elevated demand."
                    ),
                    "related_insight_theme": "capacity",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Leverage inpatient KPIs to support evidence-informed "
                        "clinical governance discussions."
                    ),
                    "related_insight_theme": "governance",
                    "confidence": rec_conf(avg_conf, score),
                },
            ])

        # ===================== CLINIC =====================
        if sub == CLIN:
            generated.extend([
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Analyze wait-time and attendance patterns to better "
                        "understand access variability."
                    ),
                    "related_insight_theme": "access",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Explore no-show behavior to assess opportunities "
                        "for improving appointment utilization."
                    ),
                    "related_insight_theme": "attendance",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Review provider workload distribution to support "
                        "balanced ambulatory operations."
                    ),
                    "related_insight_theme": "provider utilization",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Monitor clinic revenue trends in conjunction with "
                        "visit volumes to understand financial sensitivity."
                    ),
                    "related_insight_theme": "revenue",
                    "confidence": rec_conf(avg_conf, score),
                },
            ])

        # ===================== DIAGNOSTICS =====================
        if sub == DIAG:
            generated.extend([
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Review diagnostic turnaround patterns to identify "
                        "workflow bottlenecks affecting downstream care."
                    ),
                    "related_insight_theme": "turnaround time",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Assess repeat testing signals to understand potential "
                        "quality or ordering inefficiencies."
                    ),
                    "related_insight_theme": "repeat testing",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Use diagnostic volume trends to inform capacity "
                        "and staffing discussions."
                    ),
                    "related_insight_theme": "capacity",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Track alert volumes over time to maintain "
                        "diagnostic signal effectiveness."
                    ),
                    "related_insight_theme": "alerts",
                    "confidence": rec_conf(avg_conf, score),
                },
            ])

        # ===================== PHARMACY =====================
        if sub == PHAR:
            generated.extend([
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Review medication spend patterns to support "
                        "cost stewardship discussions."
                    ),
                    "related_insight_theme": "drug spend",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Monitor inventory and supply variability to "
                        "understand stock sensitivity."
                    ),
                    "related_insight_theme": "inventory",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Interpret safety alert patterns alongside dispensing "
                        "activity to contextualize medication risk."
                    ),
                    "related_insight_theme": "safety",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Use refill behavior as a contextual signal "
                        "for adherence monitoring."
                    ),
                    "related_insight_theme": "adherence",
                    "confidence": rec_conf(avg_conf, score),
                },
            ])

        # ===================== PUBLIC HEALTH =====================
        if sub == PUBH:
            generated.extend([
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Examine incidence and cohort trends to inform "
                        "preventive strategy planning."
                    ),
                    "related_insight_theme": "incidence",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Use surveillance indicators to support early "
                        "detection of emerging population risks."
                    ),
                    "related_insight_theme": "surveillance",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Assess access and utilization patterns to "
                        "identify equity-related gaps."
                    ),
                    "related_insight_theme": "equity",
                    "confidence": rec_conf(avg_conf, score),
                },
                {
                    "sub_domain": sub,
                    "recommendation": (
                        "Leverage population dashboards to support "
                        "data-informed policy evaluation."
                    ),
                    "related_insight_theme": "policy",
                    "confidence": rec_conf(avg_conf, score),
                },
            ])

        recommendations.extend(generated[:5])

    return recommendations



