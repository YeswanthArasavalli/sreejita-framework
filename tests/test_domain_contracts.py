"""
CI guard tests for v1.x public domain contracts.

These tests ensure backward compatibility and
structural stability during stabilization mode.

RULES:
- No domain logic execution
- No confidence assumptions
- No intelligence validation
- Imports + structural contracts ONLY
"""


# =====================================================
# DOMAIN DETECTOR CONTRACTS
# =====================================================

def test_domain_detectors_exist():
    """
    Ensure all public domain detectors remain importable.
    """

    # Retail
    from sreejita.domains.retail import RetailDomainDetector

    # Customer
    from sreejita.domains.customer import CustomerDomainDetector

    # Finance
    from sreejita.domains.finance import FinanceDomainDetector

    # Healthcare is optional and must not break CI
    try:
        from sreejita.domains.healthcare import HealthcareDomainDetector
    except ImportError:
        pass


# =====================================================
# DOMAIN CLASS CONTRACTS
# =====================================================

def test_domain_classes_exist_and_expose_run():
    """
    Ensure all public domain classes:
    - are importable
    - expose a callable run() method

    No execution is performed.
    """

    from sreejita.domains.retail import RetailDomain
    from sreejita.domains.customer import CustomerDomain
    from sreejita.domains.finance import FinanceDomain

    for domain_cls in (RetailDomain, CustomerDomain, FinanceDomain):
        assert hasattr(domain_cls, "run"), f"{domain_cls.__name__} missing run()"
        assert callable(getattr(domain_cls, "run")), f"{domain_cls.__name__}.run is not callable"
