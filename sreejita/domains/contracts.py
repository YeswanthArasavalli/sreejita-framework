# =====================================================
# DOMAIN CONTRACTS — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6
# =====================================================

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# =====================================================
# BASE DOMAIN DETECTOR (AUTHORITATIVE)
# =====================================================

class BaseDomainDetector:
    """
    Base class for all domain detectors.

    CONTRACT (v3.6):
    - detect(df) MUST return DomainDetectionResult
    - Detector MUST NOT attach execution engine
    - Detector MUST NOT raise (router guards, but detectors should be safe)
    - Detector MUST be stateless
    """

    domain_name: str = "generic"

    def detect(self, df) -> "DomainDetectionResult":
        """
        Detect whether the dataset belongs to this domain.

        MUST:
        - Never raise
        - Always return DomainDetectionResult
        """
        raise NotImplementedError(
            "Domain detectors must implement detect(df)"
        )


# =====================================================
# DOMAIN DETECTION RESULT (AUTHORITATIVE CONTRACT)
# =====================================================

@dataclass
class DomainDetectionResult:
    """
    Domain Detection Result — Authoritative Contract

    Represents:
    - WHY a domain matched (signals)
    - HOW confident the detector is

    Execution engine is attached LATER by router/orchestrator.
    """

    domain: Optional[str]
    confidence: float
    signals: Dict[str, Any] = field(default_factory=dict)

    # Execution handle (router-attached ONLY)
    engine: Optional[Any] = field(default=None, repr=False)

    # -------------------------------------------------
    # POST-INIT NORMALIZATION (CRITICAL & NON-NEGOTIABLE)
    # -------------------------------------------------
    def __post_init__(self):
        # -------------------------------
        # DOMAIN NORMALIZATION
        # -------------------------------
        if not isinstance(self.domain, str) or not self.domain.strip():
            self.domain = None
        else:
            self.domain = self.domain.strip().lower()

        # -------------------------------
        # CONFIDENCE NORMALIZATION
        # -------------------------------
        try:
            conf = float(self.confidence)
        except Exception:
            conf = 0.0

        # Clamp confidence to [0.0, 1.0]
        self.confidence = max(0.0, min(conf, 1.0))

        # -------------------------------
        # SIGNAL NORMALIZATION
        # -------------------------------
        if not isinstance(self.signals, dict):
            self.signals = {}

        # -------------------------------
        # ENGINE SAFETY (ABSOLUTE RULE)
        # -------------------------------
        # Detectors must NEVER attach engines.
        # Router owns this lifecycle exclusively.
        if self.engine is not None:
            self.engine = None
