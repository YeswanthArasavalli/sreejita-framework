import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# =====================================================
# Phase-3 safety thresholds (deterministic and explicit)
# =====================================================

_MIN_SAMPLE_SIZE = 20
_MIN_SIGNAL_STRENGTH = 0.40


# =====================================================
# Safety helpers
# =====================================================

def _safe_score(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(out) or out in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(out, 1.0))


def _resolve_signal_strength(df: pd.DataFrame) -> Tuple[bool, float, str]:
    """
    Phase-3 rule:
    - Visuals do NOT infer signal
    - They may consume explicit upstream signal
    """
    if "signal_strength" not in df.attrs:
        return False, 0.0, "missing_signal_strength"

    signal_strength = _safe_score(df.attrs.get("signal_strength"))
    if signal_strength < _MIN_SIGNAL_STRENGTH:
        return False, signal_strength, "weak_signal_strength"

    return True, signal_strength, "ok"


def _has_zero_variance(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return True
    if clean.nunique(dropna=True) <= 1:
        return True
    return clean.std() == 0.0


def _suppression_metadata(
    reason: str,
    confidence: float,
    sample_size: int,
    signal_strength: float,
) -> Dict[str, Any]:
    return {
        "status": "insufficient_data",
        "reason": reason,
        "confidence": _safe_score(confidence),
        "sample_size": int(sample_size),
        "signal_strength": _safe_score(signal_strength),
        "inference_type": "suppressed",
    }


def _write_visual_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _render_insufficient_data_visual(
    path: Path,
    title: str,
    metadata: Dict[str, Any],
) -> None:
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.title(title)

    text = (
        "INSUFFICIENT DATA\n\n"
        f"Reason: {metadata['reason']}\n"
        f"Signal Strength: {metadata['signal_strength']:.2f}\n"
        f"Sample Size: {metadata['sample_size']}"
    )

    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =====================================================
# Visuals
# =====================================================

def shipping_cost_vs_sales(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "shipping_cost_vs_sales.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = len(df)
    ok_signal, signal_strength, signal_reason = _resolve_signal_strength(df)

    # 1️⃣ Missing / weak signal dominates everything
    if not ok_signal:
        meta = _suppression_metadata(
            reason=signal_reason,
            confidence=signal_strength,
            sample_size=sample_size,
            signal_strength=signal_strength,
        )
        _render_insufficient_data_visual(path, "Shipping Cost vs Sales", meta)
        _write_visual_metadata(path, meta)
        return path

    # 2️⃣ Zero variance
    if _has_zero_variance(df["sales"]) or _has_zero_variance(df["shipping_cost"]):
        meta = _suppression_metadata(
            reason="zero_variance",
            confidence=signal_strength,
            sample_size=sample_size,
            signal_strength=signal_strength,
        )
        _render_insufficient_data_visual(path, "Shipping Cost vs Sales", meta)
        _write_visual_metadata(path, meta)
        return path

    # 3️⃣ Sample size
    if sample_size < _MIN_SAMPLE_SIZE:
        meta = _suppression_metadata(
            reason="sample_size_below_minimum",
            confidence=signal_strength,
            sample_size=sample_size,
            signal_strength=signal_strength,
        )
        _render_insufficient_data_visual(path, "Shipping Cost vs Sales", meta)
        _write_visual_metadata(path, meta)
        return path

    # ✅ Render
    plt.figure(figsize=(6, 4))
    plt.scatter(df["sales"], df["shipping_cost"], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    _write_visual_metadata(
        path,
        {
            "status": "rendered",
            "reason": "ok",
            "confidence": signal_strength,
            "sample_size": sample_size,
            "signal_strength": signal_strength,
            "inference_type": "direct",
        },
    )

    return path


def discount_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "discount_distribution.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = len(df)
    ok_signal, signal_strength, signal_reason = _resolve_signal_strength(df)

    if not ok_signal:
        meta = _suppression_metadata(
            reason=signal_reason,
            confidence=signal_strength,
            sample_size=sample_size,
            signal_strength=signal_strength,
        )
        _render_insufficient_data_visual(path, "Distribution of Discounts", meta)
        _write_visual_metadata(path, meta)
        return path

    if _has_zero_variance(df["discount"]):
        meta = _suppression_metadata(
            reason="zero_variance",
            confidence=signal_strength,
            sample_size=sample_size,
            signal_strength=signal_strength,
        )
        _render_insufficient_data_visual(path, "Distribution of Discounts", meta)
        _write_visual_metadata(path, meta)
        return path

    if sample_size < _MIN_SAMPLE_SIZE:
        meta = _suppression_metadata(
            reason="sample_size_below_minimum",
            confidence=signal_strength,
            sample_size=sample_size,
            signal_strength=signal_strength,
        )
        _render_insufficient_data_visual(path, "Distribution of Discounts", meta)
        _write_visual_metadata(path, meta)
        return path

    plt.figure(figsize=(6, 4))
    plt.hist(df["discount"], bins=20, alpha=0.7)
    plt.xlabel("Discount Rate")
    plt.ylabel("Frequency")
    plt.title("Distribution of Discounts")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    _write_visual_metadata(
        path,
        {
            "status": "rendered",
            "reason": "ok",
            "confidence": signal_strength,
            "sample_size": sample_size,
            "signal_strength": signal_strength,
            "inference_type": "direct",
        },
    )

    return path
