# =====================================================
# ORCHESTRATOR — UNIVERSAL (AUTHORITATIVE)
# Sreejita Framework v3.6.2 (LOCKED, STABILIZED)
# =====================================================

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List

import pandas as pd

from sreejita.domains.router_v2 import detect_domain
from sreejita.domains.registry import registry

from sreejita.reporting.storytelling_layer import apply_storytelling_layer
from sreejita.narrative.executive_cognition import (
    build_executive_payload,
    build_subdomain_executive_payloads,
)

from sreejita.reporting.recommendation_enricher import enrich_recommendations
from sreejita.core.dataset_shape import detect_dataset_shape
from sreejita.core.fingerprint import dataframe_fingerprint

log = logging.getLogger("sreejita.orchestrator")

# =====================================================
# GOVERNANCE CONSTANTS (LOCKED)
# =====================================================

MIN_DOMAIN_CONFIDENCE = 0.40
MAX_EXECUTIVE_VISUALS = 6


# =====================================================
# CONFIDENCE & EVIDENCE SAFETY HELPERS
# =====================================================

def _safe_confidence(value: Any) -> Optional[float]:
    """Return a bounded confidence if finite; otherwise None (no silent defaults)."""
    try:
        out = float(value)
    except Exception:
        return None
    if pd.isna(out) or out in (float("inf"), float("-inf")):
        return None
    return max(0.0, min(out, 1.0))


def _min_confidence(values: Iterable[Any]) -> Optional[float]:
    safe_vals = [_safe_confidence(v) for v in values]
    safe_vals = [v for v in safe_vals if v is not None]
    if not safe_vals:
        return None
    return min(safe_vals)


def _extract_confidence(item: Any) -> Optional[float]:
    if isinstance(item, dict) and "confidence" in item:
        return _safe_confidence(item.get("confidence"))
    return None


def _extract_signal_strength(item: Any) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    if "signal_strength" in item:
        return _safe_confidence(item.get("signal_strength"))
    if isinstance(item.get("kpi"), dict):
        return _safe_confidence(item["kpi"].get("signal_strength"))
    return None


def _extract_data_coverage(item: Any) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    if "data_coverage" in item:
        return _safe_confidence(item.get("data_coverage"))
    if isinstance(item.get("kpi"), dict):
        return _safe_confidence(item["kpi"].get("data_coverage"))
    return None


def _is_insufficient_component(item: Any) -> bool:
    return isinstance(item, dict) and item.get("status") == "insufficient_data"


def _recommendation_has_evidence(rec: Any) -> bool:
    if not isinstance(rec, dict):
        return False
    evidence_keys = ("evidence", "kpi", "kpis", "signal_strength", "data_coverage")
    return any(k in rec and rec.get(k) not in (None, "", [], {}) for k in evidence_keys)


# =====================================================
# SAFE FILE LOADER
# =====================================================

def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for enc in (None, "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue

    if path.suffix.lower() in (".xls", ".xlsx"):
        for engine in (None, "openpyxl", "xlrd"):
            try:
                return pd.read_excel(path, engine=engine)
            except Exception:
                continue

    raise RuntimeError(f"Unsupported file type: {path.suffix}")


# =====================================================
# BOARD READINESS HISTORY (NON-BLOCKING)
# =====================================================

def _load_history(run_dir: Path) -> Dict[str, int]:
    try:
        path = run_dir / "board_history.json"
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_history(run_dir: Path, history: Dict[str, int]) -> None:
    try:
        path = run_dir / "board_history.json"
        path.write_text(json.dumps(history, indent=2))
    except Exception:
        pass


def _trend(prev: Optional[int], curr: Optional[int]) -> Optional[str]:
    if prev is None or curr is None:
        return None
    if curr > prev:
        return "up"
    if curr < prev:
        return "down"
    return "flat"


# =====================================================
# REPORT ORCHESTRATION (AUTHORITATIVE)
# =====================================================

def generate_report_payload(
    path: Path,
    run_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    config = config or {}

    # -------------------------------------------------
    # 1. LOAD DATASET
    # -------------------------------------------------
    raw_df = _read_tabular_file_safe(path)
    if raw_df.empty:
        raise RuntimeError("Dataset is empty")

    df = raw_df.copy(deep=False)
    dataset_key = dataframe_fingerprint(df)

    # -------------------------------------------------
    # 2. DATASET SHAPE (CONTEXT ONLY)
    # -------------------------------------------------
    shape_info = detect_dataset_shape(df)

    # -------------------------------------------------
    # 3. DOMAIN DETECTION (AUTHORITATIVE)
    # -------------------------------------------------
    detection = detect_domain(
        df,
        domain_hint=config.get("domain_hint"),
        strict=False,
    )

    # =================================================
    # 🚨 NO DOMAIN — SUPPRESSED REPORT
    # =================================================
    if not detection or not detection.domain:
        det_conf = _safe_confidence(detection.confidence if detection else 0.0) or 0.0
        return {
            "unknown": {
                "domain": None,
                "confidence": det_conf,
                "status": "insufficient_data",
                "kpis": {},
                "visuals": [],
                "insights": [],
                "recommendations": [],
                "executive": {
                    "executive_brief": (
                        "The dataset does not contain sufficient domain-specific "
                        "signals to confidently classify it into a known business domain."
                    ),
                    "board_readiness": {"score": None, "band": "Insufficient Data"},
                    "limitations": [
                        "Domain classification confidence below acceptable threshold",
                        "Key domain-specific signals were missing or ambiguous",
                    ],
                },
                "shape": shape_info,
            }
        }

    domain = detection.domain
    confidence = _safe_confidence(detection.confidence)

    if confidence is None:
        return {
            domain: {
                "domain": domain,
                "confidence": 0.0,
                "status": "insufficient_data",
                "kpis": {},
                "visuals": [],
                "insights": [],
                "recommendations": [],
                "executive": {
                    "executive_brief": (
                        f"The dataset maps to '{domain}', but decision confidence is unavailable "
                        "so execution is suppressed for integrity."
                    ),
                    "board_readiness": {"score": None, "band": "Insufficient Data"},
                    "limitations": ["Missing or non-finite domain confidence"],
                },
                "shape": shape_info,
            }
        }

    # =================================================
    # 🚨 LOW CONFIDENCE — EXECUTION SUPPRESSED
    # =================================================
    if confidence < MIN_DOMAIN_CONFIDENCE:
        return {
            domain: {
                "domain": domain,
                "confidence": confidence,
                "status": "ambiguous",
                "kpis": {},
                "visuals": [],
                "insights": [],
                "recommendations": [],
                "executive": {
                    "executive_brief": (
                        f"The dataset weakly resembles the '{domain}' domain, "
                        "but confidence is insufficient for reliable analysis."
                    ),
                    "board_readiness": {"score": None, "band": "Low Confidence"},
                    "limitations": [
                        "Domain confidence below minimum execution threshold"
                    ],
                },
                "shape": shape_info,
            }
        }

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION
    # -------------------------------------------------
    try:
        engine = registry[domain]
        result = engine.run(
            df,
            visual_output_dir=run_dir / "visuals" / domain,
        )

        kpis = result.get("kpis", {}) or {}
        visuals = result.get("visuals", []) or []
        insights = result.get("insights", []) or []
        recommendations = result.get("recommendations", []) or []

        insights = apply_storytelling_layer(
            insights=insights,
            kpis=kpis,
            df=df,
            domain=domain,
        )

        recommendations = enrich_recommendations(recommendations)

    except Exception as e:
        log.exception(f"Domain execution failed | domain={domain}")
        return {
            domain: {
                "domain": domain,
                "confidence": 0.0,
                "status": "insufficient_data",
                "kpis": {},
                "visuals": [],
                "insights": [],
                "recommendations": [],
                "executive": {
                    "executive_brief": (
                        f"Analysis for domain '{domain}' could not be completed "
                        "due to an internal processing error."
                    ),
                    "limitations": [str(e)],
                },
                "shape": shape_info,
            }
        }

    # -------------------------------------------------
    # 5. CONFIDENCE-PRESERVING OUTPUT GATES
    # -------------------------------------------------
    limitations: List[str] = []

    gated_insights = []
    for insight in insights:
        ss = _extract_signal_strength(insight)
        if ss is None or ss < MIN_DOMAIN_CONFIDENCE:
            limitations.append("Suppressed insight due to weak or missing signal_strength")
            continue
        gated_insights.append(insight)
    insights = gated_insights

    gated_recommendations = []
    for rec in recommendations:
        if not _recommendation_has_evidence(rec):
            limitations.append("Suppressed recommendation without evidence linkage")
            continue
        ss = _extract_signal_strength(rec)
        if ss is not None and ss < MIN_DOMAIN_CONFIDENCE:
            limitations.append("Suppressed recommendation below signal_strength threshold")
            continue
        gated_recommendations.append(rec)
    recommendations = gated_recommendations

    # -------------------------------------------------
    # 6. EXECUTIVE VISUAL SELECTION
    # -------------------------------------------------
    visuals = sorted(
        visuals,
        key=lambda v: float(v.get("importance", 0.0)) * float(v.get("confidence", 0.0)),
        reverse=True,
    )[:MAX_EXECUTIVE_VISUALS]

    # -------------------------------------------------
    # 7. EXECUTIVE COGNITION
    # -------------------------------------------------
    executive = build_executive_payload(
        kpis=kpis,
        insights=insights,
        recommendations=recommendations,
        domain=domain,
    )

    executive["sub_domains"] = build_subdomain_executive_payloads(
        kpis,
        insights,
        recommendations,
        domain=domain,
    )

    # -------------------------------------------------
    # 8. END-TO-END CONFIDENCE PROPAGATION (MIN-DOMINANT)
    # -------------------------------------------------
    conf_candidates = [confidence]

    for group in (visuals, insights, recommendations):
        for item in group:
            c = _extract_confidence(item)
            if c is not None:
                conf_candidates.append(c)

    for kpi_val in kpis.values():
        c = _extract_confidence(kpi_val)
        if c is not None:
            conf_candidates.append(c)

    aggregate_conf = _min_confidence(conf_candidates)
    executive["confidence"] = aggregate_conf

    if aggregate_conf is None:
        return {
            domain: {
                "domain": domain,
                "confidence": 0.0,
                "status": "insufficient_data",
                "kpis": kpis,
                "visuals": [],
                "insights": [],
                "recommendations": [],
                "executive": {
                    "executive_brief": (
                        f"Analysis for '{domain}' was suppressed because aggregate confidence "
                        "could not be computed from finite component confidences."
                    ),
                    "board_readiness": {"score": None, "band": "Insufficient Data"},
                    "limitations": limitations + [
                        "Missing finite confidence in contributing elements"
                    ],
                },
                "shape": shape_info,
            }
        }

    has_insufficient_component = (
        any(_is_insufficient_component(i) for i in insights)
        or any(_is_insufficient_component(r) for r in recommendations)
        or _is_insufficient_component(executive)
    )

    if has_insufficient_component:
        aggregate_status = "insufficient_data"
    elif aggregate_conf < MIN_DOMAIN_CONFIDENCE:
        aggregate_status = "ambiguous"
    else:
        aggregate_status = "detected"

    if aggregate_status != "detected":
        if insights:
            limitations.append("Suppressed insights due to aggregate confidence downgrade")
        if recommendations:
            limitations.append("Suppressed recommendations due to aggregate confidence downgrade")
        insights = []
        recommendations = []

    # -------------------------------------------------
    # 9. BOARD READINESS TREND
    # -------------------------------------------------
    history = _load_history(run_dir)

    board = executive.get("board_readiness", {}) or {}
    current_score = board.get("score")
    previous_score = history.get(dataset_key)

    executive["board_readiness_trend"] = {
        "previous_score": previous_score,
        "current_score": current_score,
        "trend": _trend(previous_score, current_score),
    }

    if limitations:
        executive["limitations"] = executive.get("limitations", []) + limitations

    if isinstance(current_score, int):
        history[dataset_key] = current_score
        _save_history(run_dir, history)

    # -------------------------------------------------
    # 10. FINAL PAYLOAD (SINGLE DOMAIN)
    # -------------------------------------------------
    return {
        domain: {
            "domain": domain,
            "confidence": aggregate_conf,
            "status": aggregate_status,
            "kpis": kpis,
            "visuals": visuals,
            "insights": insights,
            "recommendations": recommendations,
            "executive": executive,
            "shape": shape_info,
        }
    }
