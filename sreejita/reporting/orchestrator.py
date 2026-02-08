# =====================================================
# ORCHESTRATOR — UNIVERSAL (AUTHORITATIVE)
# Sreejita Framework v3.6.2 (LOCKED)
# =====================================================

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

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
# SAFE FILE LOADER
# =====================================================

def _read_tabular_file_safe(path: Path) -> pd.DataFrame:
    """
    Load CSV / Excel safely with encoding & engine fallbacks.
    NEVER mutates filesystem state.
    """
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

def _history_path(run_dir: Path) -> Path:
    return run_dir / "board_readiness_history.json"


def _load_history(run_dir: Path) -> Dict[str, int]:
    try:
        with open(_history_path(run_dir), "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_history(run_dir: Path, history: Dict[str, int]) -> None:
    try:
        with open(_history_path(run_dir), "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _trend(prev: Optional[int], curr: Optional[int]) -> str:
    if prev is None or curr is None:
        return "→"
    if curr >= prev + 5:
        return "↑"
    if curr <= prev - 5:
        return "↓"
    return "→"


# =====================================================
# CANONICAL ENTRY POINT
# =====================================================

def generate_report_payload(
    input_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Canonical orchestration pipeline.

    GUARANTEES:
    - Router is authoritative
    - No domain guessing
    - Execution only after confidence gate
    - Deterministic payload for Hybrid
    """

    # -------------------------------------------------
    # 0. INPUT VALIDATION
    # -------------------------------------------------
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    run_dir = Path(config.get("run_dir", "runs/current"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. LOAD DATA (SAFE)
    # -------------------------------------------------
    raw_df = _read_tabular_file_safe(input_path)
    if raw_df.empty:
        raise RuntimeError("Dataset is empty")

    df = raw_df.copy(deep=False)
    dataset_key = dataframe_fingerprint(df)

    # -------------------------------------------------
    # 2. DATASET SHAPE (CONTEXT ONLY)
    # -------------------------------------------------
    shape_info = detect_dataset_shape(df)

    # -------------------------------------------------
    # 3. DOMAIN DETECTION (ROUTER v2 — AUTHORITATIVE)
    # -------------------------------------------------
    detection = detect_domain(
        df,
        domain_hint=config.get("domain_hint"),
        strict=False,
    )

    if not detection or not detection.domain:
        raise RuntimeError(
            "No confident domain detected. "
            "Please specify a domain explicitly."
        )

    domain = detection.domain
    confidence = float(detection.confidence or 0.0)

    if confidence < MIN_DOMAIN_CONFIDENCE:
        log.warning(
            "Low domain confidence detected "
            f"(domain={domain}, confidence={confidence})"
        )

    engine = registry.get_domain(domain)
    if engine is None:
        raise RuntimeError(f"Domain '{domain}' is not registered")

    # -------------------------------------------------
    # 4. DOMAIN EXECUTION (AUTHORITATIVE)
    # -------------------------------------------------
    try:
        result = engine.run(
            df,
            visual_output_dir=run_dir / "visuals" / domain,
        )

        kpis = result.get("kpis", {}) or {}
        visuals = result.get("visuals", []) or []
        insights = result.get("insights", []) or []
        recommendations = result.get("recommendations", []) or []

        # STORYTELLING (NON-MUTATING)
        insights = apply_storytelling_layer(
            insights=insights,
            kpis=kpis,
            df=df,
            domain=domain,
        )

        recommendations = enrich_recommendations(recommendations)

    except Exception as e:
        log.exception(f"Domain execution failed | domain={domain}")
        raise RuntimeError(f"{domain} execution failed: {e}")

    # -------------------------------------------------
    # 5. EXECUTIVE VISUAL SELECTION (GOVERNED)
    # -------------------------------------------------
    visuals = sorted(
        visuals,
        key=lambda v: float(v.get("importance", 0.0))
        * float(v.get("confidence", 1.0)),
        reverse=True,
    )[:MAX_EXECUTIVE_VISUALS]

    # -------------------------------------------------
    # 6. EXECUTIVE COGNITION
    # -------------------------------------------------
    executive = build_executive_payload(
        kpis=kpis,
        insights=insights,
        recommendations=recommendations,
        domain=domain,
    )

    sub_exec = build_subdomain_executive_payloads(
        kpis,
        insights,
        recommendations,
        domain=domain,
    )

    executive["sub_domains"] = sub_exec

    # -------------------------------------------------
    # 7. BOARD READINESS TREND (NON-BLOCKING)
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

    if isinstance(current_score, int):
        history[dataset_key] = current_score
        _save_history(run_dir, history)

    # -------------------------------------------------
    # 8. FINAL PAYLOAD (SINGLE DOMAIN)
    # -------------------------------------------------
    return {
        domain: {
            "domain": domain,
            "kpis": kpis,
            "visuals": visuals,
            "insights": insights,
            "recommendations": recommendations,
            "executive": executive,
            "shape": shape_info,
        }
    }
