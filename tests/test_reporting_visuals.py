import json
from pathlib import Path

import pandas as pd

from sreejita.reporting.visuals import shipping_cost_vs_sales, discount_distribution


def _read_meta(path: Path):
    return json.loads(path.with_suffix('.json').read_text())


def test_shipping_visual_suppressed_when_signal_missing(tmp_path):
    df = pd.DataFrame({"sales": list(range(12)), "shipping_cost": [v * 0.2 for v in range(12)]})
    path = shipping_cost_vs_sales(df, tmp_path)

    assert path.exists()
    meta = _read_meta(path)
    assert meta["status"] == "insufficient_data"
    assert meta["reason"] == "missing_signal_strength"


def test_discount_visual_suppressed_on_zero_variance(tmp_path):
    df = pd.DataFrame({"discount": [0.1] * 12})
    df.attrs["signal_strength"] = 0.9

    path = discount_distribution(df, tmp_path)
    assert path.exists()
    meta = _read_meta(path)
    assert meta["status"] == "insufficient_data"
    assert meta["reason"] == "zero_variance"


def test_shipping_visual_renders_when_evidence_strong(tmp_path):
    df = pd.DataFrame({
        "sales": list(range(20)),
        "shipping_cost": [1 + (v * 0.15) for v in range(20)],
    })
    df.attrs["signal_strength"] = 0.8

    path = shipping_cost_vs_sales(df, tmp_path)
    assert path.exists()
    meta = _read_meta(path)
    assert meta["status"] == "rendered"
    assert meta["reason"] == "ok"
