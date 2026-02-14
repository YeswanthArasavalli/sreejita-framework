import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


# =====================================================
# Phase-3 governance thresholds
# =====================================================

_MIN_SAMPLE_SIZE = 20


# =====================================================
# Internal helpers
# =====================================================

def _write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _render_insufficient_data_visual(
    path: Path,
    title: str,
    reason: str,
    sample_size: int,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.title(title)

    text = (
        "INSUFFICIENT DATA\n\n"
        f"Reason: {reason}\n"
        f"Sample Size: {sample_size}\n"
        "Note: Exploratory visual suppressed for executive safety"
    )

    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _has_zero_variance(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return True
    if clean.nunique(dropna=True) <= 1:
        return True
    return clean.std() == 0.0


# =====================================================
# PUBLIC API
# =====================================================

def generate_generic_visuals(
    df: pd.DataFrame,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Phase-3 Generic Visual Governance

    Rules:
    - Generic visuals are exploratory only
    - Suppressed by default under weak data
    - Never imply insight or certainty
    - Explicit uncertainty metadata required
    """
    visuals: List[Dict[str, Any]] = []
    output_dir.mkdir(exist_ok=True, parents=True)

    sample_size = int(len(df))

    # -------------------------------------------------
    # 1. Missing values overview (allowed, structural)
    # -------------------------------------------------
    missing = df.isna().sum()

    path = output_dir / "missing_values.png"

    if sample_size < _MIN_SAMPLE_SIZE or missing.empty:
        _render_insufficient_data_visual(
            path=path,
            title="Missing Values Overview",
            reason="Sample size below minimum for exploratory visualization",
            sample_size=sample_size,
        )
        _write_metadata(
            path,
            {
                "status": "insufficient_data",
                "reason": "sample_size_below_minimum",
                "inference_type": "exploratory",
                "sample_size": sample_size,
            },
        )
        visuals.append({
            "path": path,
            "caption": "Missing values overview (suppressed)",
            "status": "insufficient_data",
        })
        return visuals

    missing.plot(kind="bar", figsize=(8, 4))
    plt.title("Missing Values by Column")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    _write_metadata(
        path,
        {
            "status": "rendered",
            "inference_type": "exploratory",
            "sample_size": sample_size,
            "note": "Structural data completeness view only",
        },
    )

    visuals.append({
        "path": path,
        "caption": "Missing values across dataset columns",
        "inference_type": "exploratory",
    })

    # -------------------------------------------------
    # 2. Numeric distributions — SUPPRESSED (Phase 3)
    # -------------------------------------------------
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        col = numeric.columns[0]
        path = output_dir / "numeric_distribution.png"

        _render_insufficient_data_visual(
            path=path,
            title=f"Distribution of {col}",
            reason="Exploratory distributions suppressed in executive context",
            sample_size=sample_size,
        )

        _write_metadata(
            path,
            {
                "status": "suppressed",
                "reason": "exploratory_distribution_not_allowed",
                "inference_type": "exploratory",
                "sample_size": sample_size,
            },
        )

        visuals.append({
            "path": path,
            "caption": f"Distribution of {col} (suppressed)",
            "status": "suppressed",
        })

    # -------------------------------------------------
    # 3. Categorical rankings — SUPPRESSED (Phase 3)
    # -------------------------------------------------
    categorical = df.select_dtypes(include="object")
    if not categorical.empty:
        col = categorical.columns[0]
        path = output_dir / "top_categories.png"

        _render_insufficient_data_visual(
            path=path,
            title=f"Top Categories in {col}",
            reason="Category rankings suppressed to avoid narrative inference",
            sample_size=sample_size,
        )

        _write_metadata(
            path,
            {
                "status": "suppressed",
                "reason": "exploratory_category_ranking_not_allowed",
                "inference_type": "exploratory",
                "sample_size": sample_size,
            },
        )

        visuals.append({
            "path": path,
            "caption": f"Top categories in {col} (suppressed)",
            "status": "suppressed",
        })

    return visuals
