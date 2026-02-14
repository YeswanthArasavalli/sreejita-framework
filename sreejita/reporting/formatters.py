from typing import Optional


# =====================================================
# Phase-3 formatting helpers
# =====================================================

def _is_low_confidence(confidence: Optional[float]) -> bool:
    try:
        return confidence is not None and float(confidence) < 0.4
    except Exception:
        return False


# =====================================================
# Currency Formatter (Confidence-Aware)
# =====================================================

def fmt_currency(
    value: Optional[float],
    confidence: Optional[float] = None,
    suppressed: bool = False,
) -> str:
    """
    Canonical currency formatter (Phase 3).

    Rules:
    - Never imply false precision
    - Low confidence → approximate formatting
    - Suppressed values are explicit
    """
    if suppressed:
        return "Insufficient data"

    if value is None:
        return "—"

    try:
        value = float(value)
    except Exception:
        return "—"

    approx = _is_low_confidence(confidence)

    if abs(value) >= 1_000_000:
        formatted = f"${value/1_000_000:.1f}M" if approx else f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        formatted = f"${value/1_000:.0f}K" if approx else f"${value/1_000:.1f}K"
    else:
        formatted = f"${value:,.0f}"

    return f"~{formatted}" if approx else formatted


# =====================================================
# Percent Formatter (Confidence-Aware)
# =====================================================

def fmt_percent(
    value: Optional[float],
    confidence: Optional[float] = None,
    suppressed: bool = False,
    decimals: int = 1,
) -> str:
    """
    Canonical percent formatter (Phase 3).

    Rules:
    - Low confidence → reduced precision + approximation marker
    - Suppressed values are explicit
    """
    if suppressed:
        return "Insufficient data"

    if value is None:
        return "—"

    try:
        value = float(value)
    except Exception:
        return "—"

    approx = _is_low_confidence(confidence)

    used_decimals = 0 if approx else decimals
    formatted = f"{value * 100:.{used_decimals}f}%"

    return f"~{formatted}" if approx else formatted
