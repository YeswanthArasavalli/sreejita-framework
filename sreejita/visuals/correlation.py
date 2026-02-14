import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Any

_MIN_SAMPLE_SIZE = 20
_MIN_SIGNAL_STRENGTH = 0.40


def _safe(v: Any) -> float:
    try:
        return max(0.0, min(float(v), 1.0))
    except Exception:
        return 0.0


def _suppressed(path: Path, reason: str, df: pd.DataFrame):
    meta = {
        "status": "insufficient_data",
        "reason": reason,
        "sample_size": len(df),
        "signal_strength": _safe(df.attrs.get("signal_strength")),
        "inference_type": "suppressed",
    }

    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.title("Shipping vs Sales")
    plt.text(0.5, 0.5, f"INSUFFICIENT DATA\n\n{reason}", ha="center", va="center")
    plt.savefig(path)
    plt.close()

    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def shipping_vs_sales_visual(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shipping_vs_sales.png"

    if "signal_strength" not in df.attrs:
        return _suppressed(path, "missing_signal_strength", df)

    if _safe(df.attrs["signal_strength"]) < _MIN_SIGNAL_STRENGTH:
        return _suppressed(path, "weak_signal_strength", df)

    if len(df) < _MIN_SAMPLE_SIZE:
        return _suppressed(path, "sample_size_below_minimum", df)

    sales = next((c for c in df.columns if "sales" in c.lower()), None)
    ship = next((c for c in df.columns if "ship" in c.lower()), None)

    if not sales or not ship:
        return _suppressed(path, "missing_required_columns", df)

    if df[sales].std() == 0 or df[ship].std() == 0:
        return _suppressed(path, "zero_variance", df)

    plt.figure(figsize=(7, 4))
    plt.scatter(df[sales], df[ship], alpha=0.4)
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    meta = {
        "status": "rendered",
        "reason": "ok",
        "sample_size": len(df),
        "signal_strength": _safe(df.attrs["signal_strength"]),
        "inference_type": "direct",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path
