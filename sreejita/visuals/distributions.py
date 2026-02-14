import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

_MIN_SAMPLE_SIZE = 20


def hist(df: pd.DataFrame, col: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    if col not in df or df[col].dropna().empty:
        return None

    if len(df) < _MIN_SAMPLE_SIZE:
        return None

    if df[col].std() == 0:
        return None

    plt.figure(figsize=(5, 3))
    plt.hist(df[col].dropna(), bins=20, alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    meta = {
        "status": "rendered",
        "reason": "ok",
        "sample_size": len(df),
        "inference_type": "direct",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
