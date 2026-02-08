# =====================================================
# DECISION EXPLANATION — UNIVERSAL (STABILIZATION MODE)
# Sreejita Framework — Step 1.2
# =====================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class DecisionExplanation:
    """
    Canonical decision explanation object.

    STABILIZATION GUARANTEES:
    - Ambiguity is allowed
    - No domain is forced
    - Status explicitly declares certainty
    - Always serializable
    """

    # -------------------------------------------------
    # CORE DECISION
    # -------------------------------------------------
    decision_type: str = "domain_detection"
    selected_domain: Optional[str] = None
    confidence: float = 0.0
    status: str = "insufficient_data"  # detected | ambiguous | insufficient_data

    # -------------------------------------------------
    # EXPLAINABILITY
    # -------------------------------------------------
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
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
    # SAFETY & NORMALIZATION
    # =================================================
    def __post_init__(self):
        # -----------------------------
        # Confidence normalization
        # -----------------------------
        try:
            self.confidence = float(self.confidence)
        except Exception:
            self.confidence = 0.0

        self.confidence = max(0.0, min(self.confidence, 1.0))

        # -----------------------------
        # Container normalization
        # -----------------------------
        if not isinstance(self.alternatives, list):
            self.alternatives = []

        if not isinstance(self.signals, dict):
            self.signals = {}

        if not isinstance(self.rules_applied, list):
            self.rules_applied = []

        if self.domain_scores is not None and not isinstance(self.domain_scores, dict):
            self.domain_scores = None

        # -----------------------------
        # Core field safety (NO FORCING)
        # -----------------------------
        if not isinstance(self.decision_type, str):
            self.decision_type = "unknown_decision"

        if self.selected_domain is not None and not isinstance(self.selected_domain, str):
            self.selected_domain = None

        if self.status not in {"detected", "ambiguous", "insufficient_data"}:
            self.status = "insufficient_data"

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
