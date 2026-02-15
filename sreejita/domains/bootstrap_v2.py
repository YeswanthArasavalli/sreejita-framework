"""
Bootstrap v2 — Domain Registration (Authoritative)
Sreejita Framework v3.6

PURPOSE:
- Deterministic domain registration
- Explicit imports (NO dynamic discovery)
- Safe for CLI, UI, batch, scheduler
- Idempotent (can be imported many times)
"""

from typing import Iterable
import logging

from sreejita.domains.registry import registry

log = logging.getLogger("sreejita.bootstrap")

# =====================================================
# DOMAIN MODULE IMPORTS (EXPLICIT & ORDERED)
# =====================================================
# ⚠️ Order is for readability only — registry is order-agnostic

from sreejita.domains import (
    generic,        # 🔒 ALWAYS FIRST (fallback domain)
    retail,
    ecommerce,
    customer,
    customer_value,
    finance,
    healthcare,
    hr,
    manufacturing,
    supply_chain,
    marketing,
)

# =====================================================
# SAFE REGISTRATION HELPER
# =====================================================
def _safe_register(domain_module, registry):
    """
    Safely register a domain module.

    GUARANTEES:
    - Never overwrites existing domains
    - Structured, readable logging
    - Never raises
    """

    module_name = getattr(domain_module, "__name__", repr(domain_module))

    try:
        if not hasattr(domain_module, "register"):
            raise AttributeError(
                f"{module_name} does not expose register(registry)"
            )

        # Delegate safety to registry (authoritative)
        domain_module.register(registry)

        log.info("✅ Domain registered: %s", module_name)

    except RuntimeError as e:
        # Expected: domain already registered
        log.debug(
            "ℹ️ Domain already registered: %s (%s)",
            module_name,
            str(e),
        )

    except Exception as e:
        # Hard failure — log loudly but continue
        log.error(
            "❌ Domain registration failed: %s | %s",
            module_name,
            str(e),
            exc_info=True,
        )


# =====================================================
# BOOTSTRAP ENTRYPOINT (IDEMPOTENT)
# =====================================================
def bootstrap_domains() -> None:
    """
    Bootstrap all domain modules.

    HARD GUARANTEES:
    - Safe to call multiple times
    - Registry prevents duplicates
    - No exception ever escapes
    """

    domain_modules: Iterable = [
        generic,
        retail,
        ecommerce,
        customer,
        customer_value,
        finance,
        healthcare,
        hr,
        manufacturing,
        supply_chain,
        marketing,
    ]

    for module in domain_modules:
        _safe_register(module, registry)


# =====================================================
# AUTO-BOOTSTRAP (CRITICAL)
# =====================================================
# Ensures:
# - CLI works
# - UI works
# - Batch jobs work
# - Schedulers work
# - Re-imports are safe
bootstrap_domains()
