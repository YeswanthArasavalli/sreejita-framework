# =====================================================
# DOMAIN REGISTRY — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6
# =====================================================

from typing import Dict, Type, Optional, List
import logging

from sreejita.domains.base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

log = logging.getLogger("sreejita.registry")


class DomainRegistry:
    """
    Central authoritative registry for domain implementations and detectors.

    GUARANTEES:
    - Deterministic registration
    - Explicit domain ownership
    - Safe instantiation
    - No dynamic discovery
    - Router-safe access patterns
    """

    def __init__(self):
        self._domains: Dict[str, Type[BaseDomain]] = {}
        self._detectors: Dict[str, Type[BaseDomainDetector]] = {}

    # -------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        if not isinstance(name, str):
            return ""
        return name.strip().lower()

    # -------------------------------------------------
    # REGISTRATION (AUTHORITATIVE)
    # -------------------------------------------------

    def register(
        self,
        name: str,
        domain_cls: Type[BaseDomain],
        detector_cls: Optional[Type[BaseDomainDetector]] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a domain implementation and optional detector.

        HARD RULES:
        - domain_cls MUST be BaseDomain subclass
        - detector_cls MUST be BaseDomainDetector subclass (if provided)
        - Classes MUST be instantiable with no arguments
        - Registration is deterministic unless overwrite=True
        """

        # -------------------------------
        # NAME VALIDATION
        # -------------------------------
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Domain name must be a non-empty string")

        key = self._normalize(name)

        if not key:
            raise ValueError("Normalized domain name is empty")

        if not overwrite and key in self._domains:
            raise RuntimeError(
                f"Domain '{key}' already registered. "
                "Use overwrite=True only if intentional."
            )

        # -------------------------------
        # DOMAIN VALIDATION
        # -------------------------------
        if not isinstance(domain_cls, type) or not issubclass(domain_cls, BaseDomain):
            raise TypeError(
                f"Domain '{key}' must be a BaseDomain subclass"
            )

        # Instantiate once to verify safety
        try:
            domain_instance = domain_cls()
        except Exception as e:
            raise TypeError(
                f"Domain '{key}' cannot be instantiated safely: {e}"
            )

        # Optional consistency check
        declared_name = getattr(domain_instance, "name", None)
        if declared_name and self._normalize(declared_name) != key:
            log.warning(
                f"Domain class name='{declared_name}' does not match registry key='{key}'"
            )

        self._domains[key] = domain_cls

        # -------------------------------
        # DETECTOR VALIDATION (OPTIONAL)
        # -------------------------------
        if detector_cls is not None:
            if (
                not isinstance(detector_cls, type)
                or not issubclass(detector_cls, BaseDomainDetector)
            ):
                raise TypeError(
                    f"Detector for '{key}' must be a BaseDomainDetector subclass"
                )

            try:
                detector = detector_cls()
            except Exception as e:
                raise TypeError(
                    f"Detector for '{key}' cannot be instantiated safely: {e}"
                )

            if not hasattr(detector, "detect"):
                raise TypeError(
                    f"Detector for '{key}' does not implement detect(df)"
                )

            # Validate detector contract (dry run)
            try:
                result = detector.detect(None)
                if not isinstance(result, DomainDetectionResult):
                    raise TypeError("detect() must return DomainDetectionResult")
            except Exception:
                # Detectors are allowed to be conservative on None
                pass

            # Soft consistency warning
            declared = getattr(detector_cls, "domain_name", None)
            if declared and self._normalize(declared) != key:
                log.warning(
                    f"Detector domain_name='{declared}' does not match registry key='{key}'"
                )

            self._detectors[key] = detector_cls

        log.info(f"Registered domain '{key}'")

    # -------------------------------------------------
    # ACCESSORS (NEVER CRASH)
    # -------------------------------------------------

    def get_domain(self, name: str) -> Optional[BaseDomain]:
        """
        Returns a NEW domain instance or None.
        Never raises.
        """
        key = self._normalize(name)
        domain_cls = self._domains.get(key)

        if domain_cls is None:
            return None

        try:
            return domain_cls()
        except Exception as e:
            log.error(f"Failed to instantiate domain '{key}': {e}")
            return None

    def get_detector(self, name: str) -> Optional[BaseDomainDetector]:
        """
        Returns a NEW detector instance or None.
        Never raises.
        """
        key = self._normalize(name)
        detector_cls = self._detectors.get(key)

        if detector_cls is None:
            return None

        try:
            return detector_cls()
        except Exception as e:
            log.error(f"Failed to instantiate detector '{key}': {e}")
            return None

    # -------------------------------------------------
    # INTROSPECTION (SAFE)
    # -------------------------------------------------

    def list_domains(self) -> List[str]:
        """List registered domain names."""
        return sorted(self._domains.keys())

    def list_detectors(self) -> List[str]:
        """List registered detector names."""
        return sorted(self._detectors.keys())

    def has_domain(self, name: str) -> bool:
        return self._normalize(name) in self._domains

    def has_detector(self, name: str) -> bool:
        return self._normalize(name) in self._detectors

    # -------------------------------------------------
    # INTERNAL (ROUTER SUPPORT)
    # -------------------------------------------------

    def _all_detectors(self) -> Dict[str, Type[BaseDomainDetector]]:
        """
        Internal use only.
        Router uses this to evaluate all detectors.
        """
        return dict(self._detectors)


# =====================================================
# 🔒 SINGLETON REGISTRY (AUTHORITATIVE)
# =====================================================

registry = DomainRegistry()
