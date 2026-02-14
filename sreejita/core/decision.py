# =====================================================
# DECISION CONTRACTS — UNIVERSAL (STABILIZATION MODE)
# Sreejita Framework
# =====================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import math
import uuid


# =====================================================
# DOMAIN / EXECUTION DECISION
# =====================================================

@dataclass
class DecisionExplanation:
    """
    Canonical decision explanation object.

    STABILIZATION GUARANTEES:
    - Ambiguity is allowed
    - No domain is forced
    - Status explicitly declares certainty
    - Always serializable
    - Safe for UI, CLI, batch, and audit
    """

    # -------------------------------------------------
    # CORE DECISION
    # -------------------------------------------------
    decision_type: str = "domain_detection"
    selected_domain: Optional[str] = None
    confidence: float = 0.0
    status: str = "insufficient_data"  # detected | ambiguous | insufficient_data

    # -------------------------------------------------
    # EXPLANATION & TRACEABILITY
    # -------------------------------------------------
    alternatives: List[Any] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    rules_applied: List[str] = field(default_factory=list)

    # -------------------------------------------------
    # SCORING & TRACEABILITY
    # -------------------------------------------------
    domain_scores: Optional[Dict[str, Any]] = None
    fingerprint: Optional[str] = None

    # -------------------------------------------------
    # EXECUTION CONTEXT (NEVER SERIALIZED)
    # -------------------------------------------------
    engine: Any = field(default=None, repr=False)

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------
    decision_id: str = field(
        default_factory=lambda: f"DEC-{uuid.uuid4().hex[:10]}"
    )

    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # =================================================
    # SAFETY & NORMALIZATION (PHASE 2)
    # =================================================
    def __post_init__(self):

        def _add_rule(rule: str) -> None:
            if rule not in self.rules_applied:
                self.rules_applied.append(rule)

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        # -----------------------------
        # Confidence normalization
        # -----------------------------
        original_confidence = self.confidence
        self.confidence = max(
            0.0, min(_safe_float(self.confidence, default=0.0), 1.0)
        )

        if self.confidence != _safe_float(original_confidence, default=0.0):
            _add_rule("confidence_normalized_to_finite_range")

        # -----------------------------
        # Container normalization
        # -----------------------------
        if not isinstance(self.alternatives, list):
            self.alternatives = []
            _add_rule("alternatives_container_normalized")

        if not isinstance(self.signals, dict):
            self.signals = {}
            _add_rule("signals_container_normalized")

        if not isinstance(self.rules_applied, list):
            self.rules_applied = []
            _add_rule("rules_applied_container_normalized")

        if self.domain_scores is not None and not isinstance(self.domain_scores, dict):
            self.domain_scores = None
            _add_rule("domain_scores_container_normalized")

        # -----------------------------
        # Core field safety (NO FORCING)
        # -----------------------------
        if not isinstance(self.decision_type, str):
            self.decision_type = "unknown_decision"
            _add_rule("decision_type_normalized")

        if self.selected_domain is not None and not isinstance(self.selected_domain, str):
            self.selected_domain = None
            _add_rule("selected_domain_normalized")

        if self.status not in {"detected", "ambiguous", "insufficient_data"}:
            self.status = "insufficient_data"
            _add_rule("status_normalized")

        # -----------------------------
        # Evidence presence check
        # -----------------------------
        has_evidence = bool(self.signals or self.domain_scores or self.alternatives)

        if self.confidence > 0.0 and not has_evidence:
            self.confidence = 0.0
            self.status = "insufficient_data"
            _add_rule("downgrade_no_evidence_for_confidence")

        # -----------------------------
        # Signal strength enforcement
        # -----------------------------
        signal_strength = None

        if "signal_strength" in self.signals:
            signal_strength = _safe_float(self.signals.get("signal_strength"), None)
        elif isinstance(self.signals.get("kpi"), dict):
            signal_strength = _safe_float(
                self.signals["kpi"].get("signal_strength"), None
            )

        if signal_strength is not None and signal_strength <= 0.0:
            self.confidence = 0.0
            self.status = "insufficient_data"
            _add_rule("blocked_due_to_non_positive_signal_strength")

        # -----------------------------
        # Confidence ↔ Status consistency
        # -----------------------------
        if self.status == "detected" and self.confidence < 0.6:
            self.status = "ambiguous"
            _add_rule("downgrade_detected_due_to_low_confidence")

        if self.status == "ambiguous" and self.confidence < 0.3:
            self.status = "insufficient_data"
            _add_rule("downgrade_ambiguous_due_to_very_low_confidence")

        if self.status in {"detected", "ambiguous"} and not self.selected_domain:
            self.status = "insufficient_data"
            _add_rule("downgrade_missing_selected_domain")

        if self.status == "insufficient_data":
            self.confidence = min(self.confidence, 0.2)

    # =================================================
    # SERIALIZATION (STREAMLIT / JSON SAFE)
    # =================================================
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "selected_domain": self.selected_domain,
            "confidence": self.confidence,
            "status": self.status,
            "alternatives": self.alternatives,
            "signals": self.signals,
            "rules_applied": self.rules_applied,
            "domain_scores": self.domain_scores,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
        }

    # =================================================
    # SAFE ENGINE ATTACHMENT
    # =================================================
    def attach_engine(self, engine: Any) -> None:
        self.engine = engine

    # =================================================
    # DEBUG / LOGGING
    # =================================================
    def __str__(self) -> str:
        return (
            f"[Decision {self.decision_id}] "
            f"{self.decision_type} → {self.selected_domain} "
            f"(confidence={self.confidence:.2f}, status={self.status})"
        )

    __repr__ = __str__


# =====================================================
# POLICY DECISION (CANONICAL)
# =====================================================

@dataclass
class PolicyDecision:
    """
    Canonical policy evaluation result.

    GUARANTEES:
    - Minimal and explicit
    - Serializable
    - No hidden actions or side effects
    - Safe for UI, CLI, batch, and audit
    """

    status: str  # allowed | allowed_with_warning | blocked
    reasons: List[str] = field(default_factory=list)
