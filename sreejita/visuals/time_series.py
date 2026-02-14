import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_MIN_SAMPLE_SIZE = 6  # months, not rows


def sales_trend_visual(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sales_trend.png"

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    sales_col = next((c for c in df.columns if "sales" in c.lower()), None)

    if not date_col or not sales_col:
        return path

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    monthly = (
        df.dropna(subset=[date_col])
        .groupby(df[date_col].dt.to_period("M"))[sales_col]
        .sum()
    )

    if len(monthly) < _MIN_SAMPLE_SIZE or monthly.std() == 0:
        return path

    plt.figure(figsize=(7, 4))
    plt.plot(monthly.index.to_timestamp(), monthly.values, marker="o")
    plt.title("Sales Trend Over Time")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    meta = {
        "status": "rendered",
        "reason": "ok",
        "sample_size": len(monthly),
        "inference_type": "direct",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path
