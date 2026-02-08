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
# These are REQUIRED columns, not boosters.
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
# UNIVERSAL SUB-DOMAIN INFERENCE — HEALTHCARE (FINAL)
# =====================================================

def infer_healthcare_subdomains(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """
    Deterministic, evidence-gated healthcare sub-domain inference.

    Input MUST be resolved column map (not boolean signals).

    Returns:
    - {sub_domain: confidence}
    - Multiple entries indicate MIXED dataset
    """

    if not isinstance(cols, dict):
        return {HealthcareSubDomain.UNKNOWN.value: 1.0}

    scores: Dict[str, float] = {}

    # -------------------------------
    # HOSPITAL
    # -------------------------------
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

    # -------------------------------
    # CLINIC
    # -------------------------------
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

    # -------------------------------
    # DIAGNOSTICS
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.DIAGNOSTICS.value):
        signals = sum([
            int(_has_signal(df, cols.get("duration"))),
            int(_has_signal(df, cols.get("encounter"))),
            int(_has_signal(df, cols.get("flag"))),
        ])
        scores[HealthcareSubDomain.DIAGNOSTICS.value] = round(
            min(0.85, 0.30 + 0.15 * signals), 2
        )

    # -------------------------------
    # PHARMACY (STRICT — HARD GATED)
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PHARMACY.value):
    
        # 🚫 HARD PHARMACY GATE (NON-NEGOTIABLE)
        if not (
            cols.get("fill_date")
            and cols.get("supply")
            and cols.get("cost")
            and cols.get("fill_date") in df.columns
            and cols.get("supply") in df.columns
            and cols.get("cost") in df.columns
        ):
            pass  # 🚫 DO NOT ACTIVATE PHARMACY
        else:
            signals = sum([
                int(_has_signal(df, cols.get("fill_date"))),
                int(_has_signal(df, cols.get("supply"))),
                int(_has_signal(df, cols.get("cost"))),
            ])
    
            scores[HealthcareSubDomain.PHARMACY.value] = round(
                min(0.80, 0.35 + 0.15 * signals), 2
            )

    # -------------------------------
    # PUBLIC HEALTH
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PUBLIC_HEALTH.value):
        signals = sum([
            int(_has_signal(df, cols.get("population"))),
            int(_has_signal(df, cols.get("flag"))),
        ])
        scores[HealthcareSubDomain.PUBLIC_HEALTH.value] = round(
            min(0.90, 0.40 + 0.20 * signals), 2
        )

    if HealthcareSubDomain.CLINIC.value in scores and scores[HealthcareSubDomain.CLINIC.value] >= 0.6:
        return {HealthcareSubDomain.CLINIC.value: scores[HealthcareSubDomain.CLINIC.value]}

    # -------------------------------
    # FINAL RESOLUTION
    # -------------------------------
    if not scores:
        # If healthcare domain is confirmed but subdomain is weak,
        # assume ambulatory care by default
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

            # 🚫 No KPI → no visual
            # 🚫 No KPI evidence → no visual
            prefix = f"{sub_domain}_"
            if not any(k.startswith(prefix) for k in kpis.keys()):
                plt.close(fig)
                return

            fname = f"{sub_domain}__{visual_key}__{role}__{axis}.png"
            path = output_dir / fname
    
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    
            visual_id = f"{sub_domain}:{visual_key}:{role}"
    
            existing_ids = {
                v["visual_id"]
                for v in candidates.get(sub_domain, [])
            }
    
            if visual_id in existing_ids:
                return  # 🚫 hard dedup
    
            candidates.setdefault(sub_domain, []).append({
                "visual_id": f"{sub_domain}:{visual_key}:{role}",
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

        def driver_signature(visual_key: str, axis: str, cols: Dict[str, str]) -> str:
            """
            HARD narrative identity.
            Same column + same aggregation == SAME STORY.
            """
        
            DRIVER_MAP = {
                # TIME
                "admission_volume_trend": ("time", "count", cols.get("date")),
                "visit_volume_trend": ("time", "count", cols.get("date")),
                "dispense_volume_trend": ("time", "count", cols.get("fill_date")),
        
                "avg_los_trend": ("time", "mean", cols.get("los")),
                "ed_boarding": ("time", "mean", cols.get("duration")),
                "mortality_trend": ("time", "rate", cols.get("flag")),
                "clinic_revenue_proxy": ("time", "sum", cols.get("cost")),
                "spend_velocity": ("time", "sum", cols.get("cost")),
        
                # DISTRIBUTION
                "los_distribution": ("dist", "los", cols.get("los")),
                "wait_time_split": ("dist", "duration", cols.get("duration")),
                "inventory_turn": ("dist", "supply", cols.get("supply")),
                "cost_per_rx_distribution": ("dist", "cost", cols.get("cost")),
        
                # ENTITY
                "provider_utilization": ("entity", "doctor", cols.get("doctor")),
                "prescribing_variance": ("entity", "doctor", cols.get("doctor")),
                "order_heatmap": ("entity", "doctor_hour", cols.get("doctor")),

                # -------- CLINIC EXPERIENCE / FLOW --------
                "demographic_reach": ("dist", "facility", cols.get("facility")),
                "referral_funnel": ("entity", "facility", cols.get("facility")),
                "visit_day_pattern": ("dist", "weekday", cols.get("date")),
                "no_show_by_day": ("time", "rate", cols.get("readmitted")),
                
                # COMPOSITION
                "facility_mix": ("comp", "facility", cols.get("facility")),
                "telehealth_mix": ("comp", "admit_type", cols.get("admit_type")),
                "therapeutic_spend": ("comp", "facility", cols.get("facility")),
            }
        
            sig = DRIVER_MAP.get(visual_key)
            if not sig:
                return f"{axis}|{visual_key}"
            
            return f"{visual_key}|{axis}|{sig[-1]}"

        # -------------------------------------------------
        # VISUAL DISPATCH
        # -------------------------------------------------
        for sub in visual_subs:
    
            # KPI gate (clinic is permissive but not blind)
            if (
                sub != HealthcareSubDomain.CLINIC.value
                and not sub_has_any_kpi(sub)
            ):
                continue
    
            if sub == HealthcareSubDomain.CLINIC.value:
                if not _has_signal(df, self.cols.get("duration"), min_coverage=0.20):
                    continue
    
            # Pharmacy hard gate
            if sub == HealthcareSubDomain.PHARMACY.value:
                if not (
                    self.cols.get("fill_date") in df.columns
                    and self.cols.get("supply") in df.columns
                ):
                    continue
    
            # Public health hard gate
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                if self.cols.get("population") not in df.columns:
                    continue
    
            visual_defs = HEALTHCARE_VISUAL_MAP.get(sub, [])
            if not visual_defs:
                continue
    
            for visual_def in visual_defs:
                visual_key = visual_def["key"]
                role = visual_def["role"]
                axis = visual_def["axis"]
            
                try:
                    self._render_visual_by_key(
                        visual_key=visual_key,
                        role=role,
                        axis=axis,
                        df=df,
                        output_dir=output_dir,
                        sub_domain=sub,
                        register_visual=register_visual,
                    )
                except Exception:
                    continue

        # Ensure at least one non-time visual exists per sub-domain
        # 🚫 Remove time-only narratives early
        for sub in list(candidates.keys()):
            pool = candidates[sub]
            axes = {v["axis"] for v in pool}
            if axes == {"time"}:
                del candidates[sub]
            
        # -------------------------------------------------
        # FINAL SELECTION (MAX 6 PER SUBDOMAIN, ROLE + DRIVER BALANCED)
        # -------------------------------------------------
        ROLE_PRIORITY = [
            "volume", "flow", "quality", "financial", "utilization", "experience"
        ]
        
        for sub, pool in candidates.items():
        
            # -------------------------------------------------
            # SAFETY: DROP INVALID / LOW-CONFIDENCE VISUALS
            # -------------------------------------------------
            pool = [
                v for v in pool
                if Path(v.get("path", "")).exists()
                and v.get("confidence", 0) >= 0.35
            ]
        
            if not pool:
                continue
        
            # -------------------------------------------------
            # FIX 3 — FORCE CLINIC NON-TIME VISUALS
            # -------------------------------------------------
            if sub == HealthcareSubDomain.CLINIC.value:
                non_time = [v for v in pool if v.get("axis") != "time"]
                if len(non_time) < 2:
                    continue  # 🚫 clinic must explain flow & experience
        
            # -------------------------------------------------
            # FIX 3 — FORCE PHARMACY NON-TIME VISUALS
            # -------------------------------------------------
            if sub == HealthcareSubDomain.PHARMACY.value:
                non_time = [v for v in pool if v.get("axis") != "time"]
                if len(non_time) < 2:
                    continue  # 🚫 pharmacy must explain cost & safety

            # -------------------------------------------------
            # FIX 4 — FORCE PUBLIC HEALTH NON-TIME VISUALS
            # -------------------------------------------------
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                non_time = [v for v in pool if v.get("axis") != "time"]
                if len(non_time) < 2:
                    continue  # 🚫 public health must explain distribution & equity
        
            # -------------------------------------------------
            # DRIVER DEDUP (PREVENT SAME STORY TWICE)
            # -------------------------------------------------
            used_drivers = set()
            deduped_pool = []
        
            for v in sorted(pool, key=lambda x: -x.get("importance", 0)):
                sig = driver_signature(
                    v.get("visual_key"),
                    v.get("axis"),
                    self.cols,
                )
                if sig in used_drivers:
                    continue
                used_drivers.add(sig)
                deduped_pool.append(v)
        
            if not deduped_pool:
                continue
        
            # -------------------------------------------------
            # AXIS BALANCE ENFORCEMENT
            # -------------------------------------------------
            axis_groups = {}
            for v in deduped_pool:
                axis_groups.setdefault(v.get("axis"), []).append(v)
        
            if "time" not in axis_groups:
                continue
        
            if set(axis_groups.keys()) == {"time"}:
                continue  # 🚫 reject time-only narrative
        
            # -------------------------------------------------
            # ROLE + AXIS DIVERSITY SELECTION
            # -------------------------------------------------
            selected = []
        
            for role in ROLE_PRIORITY:
                for axis in ("time", "distribution", "composition", "entity", "correlation"):
                    role_axis_candidates = [
                        v for v in axis_groups.get(axis, [])
                        if v.get("role") == role and v.get("confidence", 0) >= 0.4
                    ]
                    if role_axis_candidates:
                        best = max(
                            role_axis_candidates,
                            key=lambda v: v.get("importance", 0) * v.get("confidence", 0),
                        )
                        selected.append(best)
                        break
        
                if len(selected) >= 6:
                    break
        
            if not selected:
                continue

            # -------------------------------------------------
            # FIX 3 — TIME AXIS MUST USE ≥2 DIFFERENT DRIVERS
            # -------------------------------------------------
            time_drivers = {
                driver_signature(v.get("visual_key"), v.get("axis"), self.cols)
                for v in selected if v.get("axis") == "time"
            }
            
            if len(time_drivers) < 2:
                non_time = [v for v in pool if v.get("axis") != "time" and v.get("confidence", 0) >= 0.4]
                time_only = [v for v in selected if v.get("axis") == "time"]
            
                # force at least 2 non-time visuals if available
                selected = []
                selected.extend(non_time[:2])
                selected.extend(time_only[:2])

            selected = selected[:6]
    
            # -------------------------------------------------
            # FIX 2 — HARD CAP TIME AXIS DOMINANCE
            # -------------------------------------------------
            time_count = sum(1 for v in selected if v.get("axis") == "time")
            if time_count > 3:
                non_time = [v for v in selected if v.get("axis") != "time"]
                time_only = [v for v in selected if v.get("axis") == "time"]
                selected = non_time + time_only[:3]
        
            # -------------------------------------------------
            # FIX 4 — HARD LIMIT DISTRIBUTION (HISTOGRAM) VISUALS
            # -------------------------------------------------
            filtered = []
            hist_count = 0
        
            for v in selected:
                if v.get("axis") == "distribution":
                    hist_count += 1
                    if hist_count > 2:
                        continue
                filtered.append(v)
        
            selected = filtered

            # -------------------------------------------------
            # FIX 2 — FACILITY / CATEGORY OVERUSE PROTECTION
            # -------------------------------------------------
            entity_count = sum(1 for v in selected if v.get("axis") == "entity")
            if entity_count > 2:
                non_entity = [v for v in selected if v.get("axis") != "entity"]
                entity_only = [v for v in selected if v.get("axis") == "entity"]
                selected = non_entity + entity_only[:2]

            # -------------------------------------------------
            # FIX — CORRELATION (SCATTER) OVERUSE PROTECTION
            # -------------------------------------------------
            corr_count = sum(1 for v in selected if v.get("axis") == "correlation")
            if corr_count > 1:
                non_corr = [v for v in selected if v.get("axis") != "correlation"]
                corr_only = [v for v in selected if v.get("axis") == "correlation"]
                selected = non_corr + corr_only[:1]
            # -------------------------------------------------
            # FIX 5 — ENSURE EXECUTIVE STORY ARC ORDER
            # -------------------------------------------------
            selected.sort(
                key=lambda v: (
                ROLE_PRIORITY.index(v["role"])
                if v.get("role") in ROLE_PRIORITY else 99
                )
            )
            
            # -------------------------------------------------
            # FINAL DRIVER DIVERSITY ENFORCEMENT (CRITICAL)
            # -------------------------------------------------
            driver_set = {
                driver_signature(v["visual_key"], v["axis"], self.cols)
                for v in selected
            }
            
            # Require at least 2 distinct drivers for a valid story
            if len(driver_set) < 2:
                # try to enrich from pool
                extras = [
                    v for v in pool
                    if driver_signature(v["visual_key"], v["axis"], self.cols) not in driver_set
                    and v.get("confidence", 0) >= 0.4
                ]
                selected.extend(extras[: (3 - len(driver_set))])

            selected = selected[:6]
        
            # -------------------------------------------------
            # PUBLISH (MAX 6)
            # -------------------------------------------------
            published.extend(selected[:6])
        
        return published

    # -------------------------------------------------
    # VISUAL RENDERER DISPATCH (REAL INTELLIGENCE)
    # -------------------------------------------------
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
        time_col = getattr(self, "time_col", None)
    
        if axis == "time":
            if time_col is None or time_col not in df.columns:
                raise ValueError("Time axis required but missing")
    
        if df is None or len(df) < 10:
            raise ValueError("Insufficient data")
    
        df = df.copy(deep=False)
        # 🚫 REQUIRED COLUMN VALIDATION
        REQUIRED_COLS = {
            "avg_los_trend": ["los"],
            "clinic_revenue_proxy": ["cost"],
            "spend_velocity": ["cost"],
            "dispense_volume_trend": ["fill_date"],
            "inventory_turn": ["supply"],
            "cost_per_rx_distribution": ["cost"],
            "hospital_revenue_proxy": ["cost"],
        }
        
        req = REQUIRED_COLS.get(visual_key)
        if req:
            for r in req:
                if not self.cols.get(r) or self.cols[r] not in df.columns:
                    raise ValueError(f"Missing required column: {r}")
    
        # =================================================
        # ---------------- HOSPITAL -----------------------
        # =================================================
        if sub_domain == HealthcareSubDomain.HOSPITAL.value:
    
            admit_col = c.get("date")
            discharge_col = c.get("discharge_date")
            los_col = c.get("los")
            cost_col = c.get("cost")
            bed_col = c.get("bed_id")
            dur_col = c.get("duration")
            flag_col = c.get("flag")
    
            if visual_key == "avg_los_trend":
                s = df[[time_col, los_col]].dropna().set_index(time_col)[los_col].resample("M").mean()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "hospital_avg_los_trend", "Average LOS trend", 0.95, 0.9, sub_domain, role, axis,)
                return

            if visual_key == "los_distribution":
                s = df[c.get("los")].dropna()
                fig, ax = plt.subplots()
                s.plot(kind="hist", bins=20, ax=ax)
                register_visual(
                    fig, "hospital_los_distribution",
                    "Length of stay distribution",
                    0.9, 0.85, sub_domain, role, axis
                )
                return

            if visual_key == "bed_turnover":
                counts = df[bed_col].dropna().value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind="hist", bins=20, ax=ax)
                register_visual(fig, "hospital_bed_turnover", "Bed turnover distribution", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "readmission_risk":
                rates = df[c.get("readmitted")].dropna().value_counts(normalize=True)
                fig, ax = plt.subplots()
                rates.plot(kind="bar", ax=ax)
                register_visual(fig, "hospital_readmission", "Readmission risk", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "discharge_hour":
                hours = df[discharge_col].dt.hour.dropna()
                fig, ax = plt.subplots()
                hours.value_counts().sort_index().plot(kind="bar", ax=ax)
                register_visual(fig, "hospital_discharge_hour", "Discharge hour distribution", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "acuity_vs_staffing":
                tmp = df[[los_col, cost_col]].dropna()
                fig, ax = plt.subplots()
                ax.scatter(tmp[los_col], tmp[cost_col], alpha=0.3)
                register_visual(fig, "hospital_acuity_staffing", "Acuity vs staffing proxy", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "ed_boarding":
                s = df[[admit_col, dur_col]].dropna().set_index(admit_col)[dur_col].resample("M").mean()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "hospital_ed_boarding", "ED boarding time trend", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "mortality_trend":
                s = df[[admit_col, flag_col]].dropna().set_index(admit_col)[flag_col].resample("M").mean()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "hospital_mortality", "Mortality proxy trend", 0.9, 0.8, sub_domain, role, axis,)
                return

            if visual_key == "facility_mix":
                fac_col = c.get("facility")
                if not fac_col or fac_col not in df.columns:
                    raise ValueError("Facility column missing")
            
                fig, ax = plt.subplots()
                df[fac_col].dropna().value_counts().plot(kind="pie", ax=ax)
                register_visual(
                    fig,
                    "hospital_facility_mix",
                    "Facility mix distribution",
                    0.8, 0.75, sub_domain, role, axis
                )
                return

            if visual_key == "admission_volume_trend":
                s = df[time_col].dropna().dt.to_period("M").value_counts().sort_index()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "hospital_volume", "Admission volume trend", 0.95, 0.9, sub_domain, role, axis,)
                return
    
            if visual_key == "hospital_revenue_proxy":
                s = df[[time_col, cost_col]].dropna().groupby(df[time_col].dt.to_period("M"))[cost_col].sum()
                fig, ax = plt.subplots()
                s.plot(kind="bar", ax=ax)
                register_visual(fig, "hospital_revenue", "Hospital cost trend (proxy)", 0.9, 0.85, sub_domain, role, axis,)
                return
    
        # =================================================
        # ---------------- CLINIC -------------------------
        # =================================================
        if sub_domain == HealthcareSubDomain.CLINIC.value:
    
            dur_col = c.get("duration")
            cost_col = c.get("cost")
            doc_col = c.get("doctor")
            flag_col = c.get("readmitted")
    
            if visual_key == "visit_volume_trend":
                s = df[time_col].dropna().dt.to_period("M").value_counts().sort_index()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "clinic_visit_volume", "Clinic visit volume trend", 0.95, 0.9, sub_domain, role, axis,)
                return
    
            if visual_key == "wait_time_split":
                fig, ax = plt.subplots()
                df[dur_col].dropna().plot(kind="hist", bins=20, ax=ax)
                register_visual(fig, "clinic_wait_time", "Clinic wait time distribution", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "appointment_lag":
                fig, ax = plt.subplots()
                df[dur_col].dropna().plot(kind="box", ax=ax)
                register_visual(fig, "clinic_cycle_time", "Visit cycle time", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "provider_utilization":
                s = df[doc_col].value_counts()
                fig, ax = plt.subplots()
                s.plot(kind="bar", ax=ax)
                register_visual(fig, "clinic_provider_util", "Provider utilization", 0.9, 0.85, sub_domain, role, axis,)
                return

            if visual_key == "visit_day_pattern":
                s = df[time_col].dt.day_name().value_counts()
                fig, ax = plt.subplots()
                s.plot(kind="bar", ax=ax)
                register_visual(
                    fig, "clinic_visit_pattern",
                    "Visit distribution by day of week",
                    0.85, 0.8, sub_domain, role, axis
                )
                return

            if visual_key == "no_show_by_day":
                s = df[[time_col, flag_col]].dropna().set_index(time_col)[flag_col].resample("M").mean()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "clinic_no_show", "No-show rate by day", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "clinic_revenue_proxy":
                s = df[[time_col, cost_col]].dropna().groupby(df[time_col].dt.to_period("M"))[cost_col].sum()
                fig, ax = plt.subplots()
                s.plot(kind="bar", ax=ax)
                register_visual(fig, "clinic_revenue", "Clinic revenue proxy", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "care_gap_proxy":
                rate = 1 - df[flag_col].dropna().mean()
                fig, ax = plt.subplots()
                ax.bar(["Care Gap Closure"], [rate])
                register_visual(fig, "clinic_care_gap", "Care gap closure proxy", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "demographic_reach":
                fig, ax = plt.subplots()
                df[c.get("facility")].value_counts().plot(kind="bar", ax=ax)
                register_visual(fig, "clinic_reach", "Clinic demographic reach", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "referral_funnel":
                fig, ax = plt.subplots()
                df[c.get("facility")].value_counts().plot(kind="bar", ax=ax)
                register_visual(fig, "clinic_referral", "Referral funnel proxy", 0.75, 0.7, sub_domain, role, axis,)
                return
    
            if visual_key == "telehealth_mix":
                fig, ax = plt.subplots()
                df[c.get("admit_type")].value_counts().plot(kind="pie", ax=ax)
                register_visual(fig, "clinic_telehealth", "Telehealth mix", 0.75, 0.7, sub_domain, role, axis,)
                return

        # =================================================
        # ---------------- DIAGNOSTICS --------------------
        # =================================================
        if sub_domain == HealthcareSubDomain.DIAGNOSTICS.value:
    
            dur_col = c.get("duration")
            flag_col = c.get("flag")
            doc_col = c.get("doctor")
            cost_col = c.get("cost")
    
            if visual_key == "order_volume_trend":
                s = df[time_col].dropna().dt.to_period("M").value_counts().sort_index()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "diag_volume", "Diagnostic order volume", 0.95, 0.9, sub_domain, role, axis,)
                return
    
            if visual_key == "tat_percentiles":
                fig, ax = plt.subplots()
                df[dur_col].dropna().plot(kind="box", ax=ax)
                register_visual(fig, "diag_tat", "Turnaround time distribution", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "critical_alert_time":
                s = df[[time_col, flag_col]].dropna().set_index(time_col)[flag_col].resample("M").mean()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "diag_alerts", "Critical alert timing", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "specimen_rejection":
                rates = df[flag_col].dropna().value_counts(normalize=True)
                fig, ax = plt.subplots()
                rates.plot(kind="bar", ax=ax)
                register_visual(fig, "diag_reject", "Specimen rejection rate", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "device_downtime":
                fig, ax = plt.subplots()
                df[dur_col].dropna().plot(kind="hist", ax=ax)
                register_visual(fig, "diag_downtime", "Device downtime proxy", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "order_heatmap":
                if doc_col not in df.columns:
                    raise ValueError("Doctor column missing")
                fig, ax = plt.subplots()
                pd.crosstab(df[doc_col], df[time_col].dt.hour).plot(ax=ax)
                register_visual(fig, "diag_heatmap", "Ordering heatmap", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "repeat_scan":
                rates = df[flag_col].dropna().value_counts(normalize=True)
                fig, ax = plt.subplots()
                rates.plot(kind="bar", ax=ax)
                register_visual(fig, "diag_repeat", "Repeat scan rate", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "test_revenue_proxy":
                s = df[[time_col, cost_col]].dropna().groupby(df[time_col].dt.to_period("M"))[cost_col].sum()
                fig, ax = plt.subplots()
                s.plot(kind="bar", ax=ax)
                register_visual(fig, "diag_revenue", "Diagnostics revenue proxy", 0.85, 0.8, sub_domain, role, axis,)
                return
    
        # =================================================
        # ---------------- PHARMACY -----------------------
        # =================================================
        if sub_domain == HealthcareSubDomain.PHARMACY.value:
            
            fill_col = c.get("fill_date")
            supply_col = c.get("supply")
            cost_col = c.get("cost")
            flag_col = c.get("flag")
    
            if visual_key == "dispense_volume_trend":
                s = df[fill_col].dropna().dt.to_period("M").value_counts().sort_index()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "pharm_volume", "Prescription volume trend", 0.95, 0.9, sub_domain, role, axis,)
                return
    
            if visual_key == "spend_velocity":
                s = df[[fill_col, cost_col]].dropna().groupby(df[fill_col].dt.to_period("M"))[cost_col].sum()
                fig, ax = plt.subplots()
                s.plot(kind="bar", ax=ax)
                register_visual(fig, "pharm_spend", "Drug spend velocity", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "therapeutic_spend":
                fac_col = c.get("facility")
                if not fac_col or fac_col not in df.columns:
                    raise ValueError("Facility column missing")
                fig, ax = plt.subplots()
                df[fac_col].value_counts().plot(kind="bar", ax=ax)
                register_visual(fig, "pharm_therapeutic", "Therapeutic class spend", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "generic_rate":
                rates = df[flag_col].dropna().value_counts(normalize=True)
                fig, ax = plt.subplots()
                rates.plot(kind="bar", ax=ax)
                register_visual(fig, "pharm_generic", "Generic substitution rate", 0.85, 0.8, sub_domain, role, axis,)
                return

            if visual_key == "cost_per_rx_distribution":
                fig, ax = plt.subplots()
                df[c.get("cost")].dropna().plot(kind="hist", bins=20, ax=ax)
                register_visual(
                    fig, "pharm_cost_dist",
                    "Cost per prescription distribution",
                    0.85, 0.8, sub_domain, role, axis
                )
                return
    
            if visual_key == "prescribing_variance":
                fig, ax = plt.subplots()
                df[c.get("doctor")].value_counts().plot(kind="bar", ax=ax)
                register_visual(fig, "pharm_variance", "Prescribing variance", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "inventory_turn":
                fig, ax = plt.subplots()
                df[supply_col].dropna().plot(kind="hist", ax=ax)
                register_visual(fig, "pharm_inventory", "Inventory turnover proxy", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "drug_alerts":
                rates = df[flag_col].dropna().value_counts(normalize=True)
                fig, ax = plt.subplots()
                rates.plot(kind="bar", ax=ax)
                register_visual(fig, "pharm_alerts", "Drug safety alerts", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "refill_gap":
                fig, ax = plt.subplots()
                df[supply_col].dropna().plot(kind="box", ax=ax)
                register_visual(fig, "pharm_refill", "Refill gap proxy", 0.8, 0.75, sub_domain, role, axis,)
                return
    
        # =================================================
        # ---------------- PUBLIC HEALTH ------------------
        # =================================================
        if sub_domain == HealthcareSubDomain.PUBLIC_HEALTH.value:
    
            pop_col = c.get("population")
            flag_col = c.get("flag")
    
            if visual_key == "population_distribution":
                fig, ax = plt.subplots()
                df[pop_col].dropna().plot(kind="hist", ax=ax)
                register_visual(fig, "ph_incidence", "Population incidence distribution", 0.95, 0.9, sub_domain, role, axis,)
                return
    
            if visual_key == "cohort_growth":
                s = df[time_col].dropna().dt.to_period("M").value_counts().sort_index()
                fig, ax = plt.subplots()
                s.plot(ax=ax)
                register_visual(fig, "ph_cohort", "Cohort growth trend", 0.9, 0.85, sub_domain, role, axis,)
                return
    
            if visual_key == "prevalence_age":
                fac_col = c.get("facility")
                if not fac_col or fac_col not in df.columns:
                    raise ValueError("Facility column missing")
                fig, ax = plt.subplots()
                df[fac_col].value_counts().plot(kind="bar", ax=ax)
                register_visual(fig, "ph_prevalence", "Prevalence by group proxy", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "access_gap":
                fig, ax = plt.subplots()
                df[flag_col].dropna().value_counts(normalize=True).plot(kind="bar", ax=ax)
                register_visual(fig, "ph_access", "Access gap proxy", 0.85, 0.8, sub_domain, role, axis,)
                return
    
            if visual_key == "program_effect":
                fig, ax = plt.subplots()
                df[flag_col].dropna().plot(kind="hist", ax=ax)
                register_visual(fig, "ph_program", "Program effect proxy", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "sdoh_overlay":
                fig, ax = plt.subplots()
                df[pop_col].dropna().plot(kind="box", ax=ax)
                register_visual(fig, "ph_sdoh", "SDOH overlay proxy", 0.8, 0.75, sub_domain, role, axis,)
                return
    
            if visual_key == "immunization_rate":
                rate = df[flag_col].dropna().mean()
                fig, ax = plt.subplots()
                ax.bar(["Immunized"], [rate])
                register_visual(fig, "ph_immunization", "Immunization rate proxy", 0.9, 0.85, sub_domain, role, axis,)
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
        # CONFIDENCE CALCULATION
        # -------------------------------------------------
        def insight_conf(kpi_conf: float, sub_score: float) -> float:
            base = min(kpi_conf or 0.0, 0.85)
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
        # CROSS-DOMAIN INSIGHTS
        # -------------------------------------------------
        hosp_score = active_subs.get(HOSP, 0.0)
        diag_score = active_subs.get(DIAG, 0.0)
    
        if hosp_score >= 0.5 and diag_score >= 0.5:
            los = self.get_kpi(kpis, HOSP, "avg_los")
            tat = self.get_kpi(kpis, DIAG, "avg_tat")
    
            if isinstance(los, (int, float)) and isinstance(tat, (int, float)) and tat > 120:
                conf_val = min(
                    self.get_kpi_confidence(kpis, HOSP, "avg_los"),
                    self.get_kpi_confidence(kpis, DIAG, "avg_tat"),
                )
                insights.append({
                    "sub_domain": "cross_domain",
                    "level": "RISK",
                    "title": "Diagnostic Delays Extending Inpatient Stay",
                    "so_what": (
                        f"Diagnostic turnaround delays ({tat:.0f} minutes) "
                        f"are associated with longer inpatient stays "
                        f"(average LOS {los:.1f} days)."
                    ),
                    "confidence": insight_conf(conf_val, min(hosp_score, diag_score)),
                })
    
        # -------------------------------------------------
        # SUB-DOMAIN COMPOSITE INSIGHTS (≥7 GENERATED)
        # -------------------------------------------------
        for sub, score in active_subs.items():
            generated: List[Dict[str, Any]] = []
    
            # ===================== HOSPITAL =====================
            if sub == HOSP and score >= 0.6:
                avg_los = self.get_kpi(kpis, sub, "avg_los")
                long_stay = self.get_kpi(kpis, sub, "long_stay_rate")
                readmit = self.get_kpi(kpis, sub, "readmission_rate")
                mort = self.get_kpi(kpis, sub, "mortality_rate")
    
                if isinstance(avg_los, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Inpatient Throughput Visibility",
                        "so_what": f"LOS is observable (avg {avg_los:.1f} days), enabling throughput governance.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "avg_los"), score),
                    })
    
                if isinstance(long_stay, (int, float)) and long_stay >= 0.25:
                    generated.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Extended Stay Concentration",
                        "so_what": f"{long_stay:.1%} of patients exceed LOS norms, stressing bed availability.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "long_stay_rate"), score),
                    })
    
                if isinstance(readmit, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Readmission Signal Detected",
                        "so_what": f"Readmission rate of {readmit:.1%} suggests post-discharge gaps.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "readmission_rate"), score),
                    })
    
                if isinstance(mort, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Mortality Proxy Observed",
                        "so_what": "Mortality proxy data indicates outcome variability requiring review.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "mortality_rate"), score),
                    })
    
                generated.append({
                    "sub_domain": sub,
                    "level": "STRENGTH",
                    "title": "Operational Data Depth",
                    "so_what": "Hospital datasets support multi-dimensional operational analysis.",
                    "confidence": insight_conf(0.7, score),
                })
    
                generated.append({
                    "sub_domain": sub,
                    "level": "WARNING",
                    "title": "Capacity Sensitivity",
                    "so_what": "High occupancy sensitivity may amplify downstream congestion.",
                    "confidence": insight_conf(0.65, score),
                })
    
                generated.append({
                    "sub_domain": sub,
                    "level": "STRENGTH",
                    "title": "Clinical Governance Readiness",
                    "so_what": "Structured inpatient KPIs enable evidence-driven governance.",
                    "confidence": insight_conf(0.7, score),
                })
    
            # ===================== CLINIC =====================
            if sub == CLIN and score >= 0.6:
                wait = self.get_kpi(kpis, sub, "avg_wait_time")
                no_show = self.get_kpi(kpis, sub, "no_show_rate")
    
                if isinstance(wait, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Clinic Access Transparency",
                        "so_what": f"Wait times measurable (avg {wait:.0f} mins), supporting scheduling optimization.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "avg_wait_time"), score),
                    })
    
                if isinstance(no_show, (int, float)) and no_show >= 0.1:
                    generated.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Elevated No-Show Risk",
                        "so_what": f"No-show rate of {no_show:.1%} reduces throughput efficiency.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "no_show_rate"), score),
                    })
    
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Ambulatory Demand Signal",
                        "so_what": "Visit volume patterns support access planning.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Revenue Sensitivity",
                        "so_what": "Clinic revenue is sensitive to attendance variability.",
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Provider Load Visibility",
                        "so_what": "Provider utilization metrics enable load balancing.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Care Continuity Gaps",
                        "so_what": "Missed follow-ups may degrade longitudinal outcomes.",
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Outpatient Analytics Readiness",
                        "so_what": "Clinic data supports proactive access and experience management.",
                        "confidence": insight_conf(0.7, score),
                    },
                ])
    
            # ===================== DIAGNOSTICS =====================
            if sub == DIAG and score >= 0.6:
                tat = self.get_kpi(kpis, sub, "avg_tat")
    
                if isinstance(tat, (int, float)):
                    generated.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Turnaround Time Visibility",
                        "so_what": f"TAT tracked (avg {tat:.0f} mins), enabling SLA enforcement.",
                        "confidence": insight_conf(self.get_kpi_confidence(kpis, sub, "avg_tat"), score),
                    })
    
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Diagnostic Throughput Pressure",
                        "so_what": "Delayed results may slow downstream clinical decisions.",
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Operational Traceability",
                        "so_what": "Diagnostic workflows are analytically traceable.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Repeat Testing Risk",
                        "so_what": "Repeat tests may indicate quality or ordering inefficiencies.",
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Utilization Insights",
                        "so_what": "Test volumes support capacity optimization.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Diagnostics Analytics Maturity",
                        "so_what": "Diagnostic KPIs enable continuous performance review.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Alert Fatigue Potential",
                        "so_what": "High alert volumes may reduce response effectiveness.",
                        "confidence": insight_conf(0.6, score),
                    },
                ])
    
            # ===================== PHARMACY =====================
            if sub == PHAR and score >= 0.6:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Medication Spend Transparency",
                        "so_what": "Drug spend visibility supports cost containment.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Inventory Sensitivity",
                        "so_what": "Supply variability may risk stock-outs.",
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Medication Safety Signals",
                        "so_what": "Alert patterns suggest potential safety exposures.",
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Pharmacy Operational Control",
                        "so_what": "Dispensing data supports workflow optimization.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Formulary Governance Potential",
                        "so_what": "Spend patterns inform formulary decisions.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Refill Adherence Risk",
                        "so_what": "Refill gaps may reduce therapy effectiveness.",
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Clinical Pharmacy Readiness",
                        "so_what": "Data supports clinical pharmacy integration.",
                        "confidence": insight_conf(0.7, score),
                    },
                ])
    
            # ===================== PUBLIC HEALTH =====================
            if sub == PUBH and score >= 0.6:
                generated.extend([
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Elevated Population Health Burden",
                        "so_what": "Incidence signals indicate prevention gaps.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Population Surveillance Coverage",
                        "so_what": "Data enables population-level monitoring.",
                        "confidence": insight_conf(0.75, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Access Inequity Signals",
                        "so_what": "Utilization disparities suggest access gaps.",
                        "confidence": insight_conf(0.65, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Program Evaluation Capability",
                        "so_what": "Intervention impact can be assessed.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Preventive Strategy Readiness",
                        "so_what": "Data supports targeted prevention planning.",
                        "confidence": insight_conf(0.7, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Delayed Response Risk",
                        "so_what": "Slow response to trends may amplify outbreaks.",
                        "confidence": insight_conf(0.6, score),
                    },
                    {
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Public Health Intelligence Maturity",
                        "so_what": "Foundational analytics capabilities are in place.",
                        "confidence": insight_conf(0.75, score),
                    },
                ])
    
            # -------------------------------------------------
            # EXPOSE MAX 5 PER SUB-DOMAIN
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
        # CONFIDENCE BINDING
        # -------------------------------------------------
        def rec_conf(ins_conf: float, sub_score: float) -> float:
            base = min(ins_conf or 0.6, 0.85)
            return round(min(0.90, base * (0.7 + 0.3 * sub_score)), 2)
    
        # -------------------------------------------------
        # SUB-DOMAIN KEYS
        # -------------------------------------------------
        HOSP = HealthcareSubDomain.HOSPITAL.value
        CLIN = HealthcareSubDomain.CLINIC.value
        DIAG = HealthcareSubDomain.DIAGNOSTICS.value
        PHAR = HealthcareSubDomain.PHARMACY.value
        PUBH = HealthcareSubDomain.PUBLIC_HEALTH.value
    
        # -------------------------------------------------
        # PER SUB-DOMAIN GENERATION (≥7)
        # -------------------------------------------------
        for sub, score in active_subs.items():
    
            sub_insights = insights_by_sub.get(sub, [])
            if not sub_insights:
                continue
    
            generated: List[Dict[str, Any]] = []
    
            for ins in sub_insights:
                level = ins.get("level")
                ins_conf = float(ins.get("confidence", 0.6))
    
                # ================= HOSPITAL =================
                if sub == HOSP:
                    generated.extend([
                        {
                            "sub_domain": sub,
                            "priority": "HIGH",
                            "action": "Implement daily multidisciplinary discharge huddles",
                            "owner": "Hospital Operations",
                            "timeline": "30–60 days",
                            "goal": "Reduce excess length of stay",
                            "expected_impact": "10–15% bed capacity improvement",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Standardize LOS benchmarks by service line",
                            "owner": "Clinical Governance",
                            "timeline": "60–90 days",
                            "goal": "Improve throughput predictability",
                            "expected_impact": "Lower LOS variability",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Introduce early discharge planning at admission",
                            "owner": "Care Coordination",
                            "timeline": "30–90 days",
                            "goal": "Prevent discharge delays",
                            "expected_impact": "Faster patient flow",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Audit readmissions for preventable causes",
                            "owner": "Quality Team",
                            "timeline": "90 days",
                            "goal": "Reduce avoidable readmissions",
                            "expected_impact": "Quality score improvement",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Align staffing levels with peak census patterns",
                            "owner": "HR / Nursing",
                            "timeline": "90–120 days",
                            "goal": "Reduce care bottlenecks",
                            "expected_impact": "Improved patient experience",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Strengthen inpatient analytics dashboards",
                            "owner": "IT / Analytics",
                            "timeline": "60 days",
                            "goal": "Operational visibility",
                            "expected_impact": "Faster management response",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Conduct quarterly throughput performance reviews",
                            "owner": "Executive Team",
                            "timeline": "Quarterly",
                            "goal": "Sustained improvement",
                            "expected_impact": "Governance maturity",
                            "confidence": rec_conf(ins_conf, score),
                        },
                    ])
    
                # ================= CLINIC =================
                if sub == CLIN:
                    generated.extend([
                        {
                            "sub_domain": sub,
                            "priority": "HIGH",
                            "action": "Deploy predictive no-show mitigation and reminders",
                            "owner": "Ambulatory Ops",
                            "timeline": "30–60 days",
                            "goal": "Reduce missed appointments",
                            "expected_impact": "5–10% visit recovery",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Rebalance appointment templates by demand",
                            "owner": "Clinic Management",
                            "timeline": "60 days",
                            "goal": "Improve access",
                            "expected_impact": "Shorter wait times",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Introduce same-day fill slots",
                            "owner": "Scheduling Team",
                            "timeline": "30–90 days",
                            "goal": "Increase throughput",
                            "expected_impact": "Higher utilization",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Expand telehealth for follow-up visits",
                            "owner": "Digital Health",
                            "timeline": "90 days",
                            "goal": "Reduce clinic congestion",
                            "expected_impact": "Better patient experience",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Monitor provider panel balance monthly",
                            "owner": "Medical Director",
                            "timeline": "Monthly",
                            "goal": "Prevent overload",
                            "expected_impact": "Sustainable workloads",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Link revenue trends to attendance patterns",
                            "owner": "Finance",
                            "timeline": "60 days",
                            "goal": "Revenue stability",
                            "expected_impact": "Improved forecasting",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Review care continuity for missed follow-ups",
                            "owner": "Care Coordination",
                            "timeline": "90 days",
                            "goal": "Improve outcomes",
                            "expected_impact": "Lower drop-offs",
                            "confidence": rec_conf(ins_conf, score),
                        },
                    ])
    
                # ================= DIAGNOSTICS =================
                if sub == DIAG:
                    generated.extend([
                        {
                            "sub_domain": sub,
                            "priority": "HIGH",
                            "action": "Enforce turnaround time SLAs with escalation",
                            "owner": "Diagnostics Head",
                            "timeline": "30–60 days",
                            "goal": "Reduce TAT",
                            "expected_impact": "Faster decisions",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Segment STAT vs routine workflows",
                            "owner": "Lab Ops",
                            "timeline": "60 days",
                            "goal": "Prioritize critical tests",
                            "expected_impact": "Reduced delays",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Audit repeat test drivers",
                            "owner": "Quality",
                            "timeline": "90 days",
                            "goal": "Reduce waste",
                            "expected_impact": "Lower costs",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Optimize staffing during peak hours",
                            "owner": "Operations",
                            "timeline": "60–90 days",
                            "goal": "Balance load",
                            "expected_impact": "Stable TAT",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Enhance ordering guidelines for clinicians",
                            "owner": "Medical Committee",
                            "timeline": "120 days",
                            "goal": "Reduce unnecessary tests",
                            "expected_impact": "Better utilization",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Track alert fatigue indicators",
                            "owner": "Quality",
                            "timeline": "Quarterly",
                            "goal": "Maintain alert effectiveness",
                            "expected_impact": "Safety improvement",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Strengthen diagnostic performance dashboards",
                            "owner": "Analytics",
                            "timeline": "60 days",
                            "goal": "Operational visibility",
                            "expected_impact": "Faster intervention",
                            "confidence": rec_conf(ins_conf, score),
                        },
                    ])
    
                # ================= PHARMACY =================
                if sub == PHAR:
                    generated.extend([
                        {
                            "sub_domain": sub,
                            "priority": "HIGH",
                            "action": "Tighten safety alert review workflows",
                            "owner": "Pharmacy Ops",
                            "timeline": "30–60 days",
                            "goal": "Reduce medication risk",
                            "expected_impact": "Lower adverse events",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Increase generic substitution monitoring",
                            "owner": "Pharmacy Leadership",
                            "timeline": "60 days",
                            "goal": "Control drug spend",
                            "expected_impact": "Cost reduction",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Review high-cost drug utilization monthly",
                            "owner": "Finance",
                            "timeline": "Monthly",
                            "goal": "Spend governance",
                            "expected_impact": "Budget control",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Strengthen inventory forecasting",
                            "owner": "Supply Chain",
                            "timeline": "60–90 days",
                            "goal": "Prevent stock-outs",
                            "expected_impact": "Supply reliability",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Monitor refill adherence for chronic meds",
                            "owner": "Clinical Pharmacy",
                            "timeline": "90 days",
                            "goal": "Improve adherence",
                            "expected_impact": "Better outcomes",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Standardize pharmacist interventions tracking",
                            "owner": "Quality",
                            "timeline": "120 days",
                            "goal": "Operational learning",
                            "expected_impact": "Process improvement",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Align formulary reviews with spend trends",
                            "owner": "Pharmacy & Finance",
                            "timeline": "Quarterly",
                            "goal": "Strategic cost control",
                            "expected_impact": "Sustained savings",
                            "confidence": rec_conf(ins_conf, score),
                        },
                    ])
    
                # ================= PUBLIC HEALTH =================
                if sub == PUBH:
                    generated.extend([
                        {
                            "sub_domain": sub,
                            "priority": "HIGH",
                            "action": "Deploy targeted prevention in high-incidence areas",
                            "owner": "Public Health Authority",
                            "timeline": "90–180 days",
                            "goal": "Reduce incidence",
                            "expected_impact": "Lower disease burden",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Improve early warning surveillance",
                            "owner": "Epidemiology Unit",
                            "timeline": "60–90 days",
                            "goal": "Faster outbreak detection",
                            "expected_impact": "Rapid response",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Address access inequities through outreach",
                            "owner": "Community Health",
                            "timeline": "120 days",
                            "goal": "Improve equity",
                            "expected_impact": "Better coverage",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Evaluate program effectiveness quarterly",
                            "owner": "Policy Team",
                            "timeline": "Quarterly",
                            "goal": "Optimize interventions",
                            "expected_impact": "Outcome improvement",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Strengthen SDOH data integration",
                            "owner": "Analytics",
                            "timeline": "180 days",
                            "goal": "Contextual insights",
                            "expected_impact": "Targeted planning",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "LOW",
                            "action": "Enhance vaccination follow-up tracking",
                            "owner": "Immunization Program",
                            "timeline": "90 days",
                            "goal": "Improve coverage",
                            "expected_impact": "Reduced outbreaks",
                            "confidence": rec_conf(ins_conf, score),
                        },
                        {
                            "sub_domain": sub,
                            "priority": "MEDIUM",
                            "action": "Establish population health dashboards",
                            "owner": "Public Health IT",
                            "timeline": "60 days",
                            "goal": "Decision support",
                            "expected_impact": "Faster policy action",
                            "confidence": rec_conf(ins_conf, score),
                        },
                    ])
    
            # -------------------------------------------------
            # EXPOSE MAX 5 PER SUB-DOMAIN
            # -------------------------------------------------
            priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            generated.sort(
                key=lambda r: (
                    priority_rank.get(r.get("priority"), 3),
                    -r.get("confidence", 0),
                )
            )
    
            recommendations.extend(generated[:5])
    
        return recommendations

# =====================================================
# HEALTHCARE DOMAIN DETECTOR (ALIAS + COVERAGE AWARE)
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    def detect(self, df: pd.DataFrame) -> DomainDetectionResult:
        """
        Universal, capability-based healthcare domain detector.

        Finance-safe guarantees:
        - Will not anchor on GL / P&L ledgers
        - Clinical signals must dominate
        """

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------
        if df is None or df.empty:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals={},
            )

        # -------------------------------------------------
        # SEMANTIC CAPABILITIES
        # -------------------------------------------------
        semantics = resolve_semantics(df)

        # -------------------------------------------------
        # 🚫 FINANCE NEGATIVE GATE (CRITICAL FIX)
        # -------------------------------------------------
        has_revenue = semantics.get("has_revenue", False)
        has_expense = semantics.get("has_expense", False)
        has_profit = semantics.get("has_profit", False)
        has_gl_account = semantics.get("has_gl_account", False)

        # Full ledger signature → NOT healthcare
        if has_revenue and has_expense and has_profit and has_gl_account:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals=semantics,
            )

        # -------------------------------------------------
        # HEALTHCARE ANCHOR (CAPABILITY-BASED)
        # -------------------------------------------------
        anchor_signals = [
            semantics.get("has_patient_id", False),
            semantics.get("has_admission_date", False),
            semantics.get("has_discharge_date", False),
            semantics.get("has_duration", False),
            semantics.get("has_cost", False),
            semantics.get("has_supply", False),
            semantics.get("has_population", False),
        ]

        anchor_score = sum(int(x) for x in anchor_signals)

        if anchor_score == 0:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals=semantics,
            )

        # -------------------------------------------------
        # CONFIDENCE (UNCHANGED)
        # -------------------------------------------------
        confidence = 0.30
        confidence += min(anchor_score * 0.10, 0.55)
        confidence = round(min(confidence, 0.95), 2)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals=semantics,
        )


def register(registry):
    registry.register(
        "healthcare",
        HealthcareDomain,
        HealthcareDomainDetector,
    )


