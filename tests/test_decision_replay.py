from sreejita.reporting.orchestrator import generate_report_payload

def test_domain_decision_is_deterministic(sample_df):
    d1 = decide_domain(sample_df)
    d2 = decide_domain(sample_df)

    # Core identity must match
    assert d1.decision_type == d2.decision_type
    assert d1.fingerprint == d2.fingerprint

    # Domain outcome must be stable (including None)
    assert d1.selected_domain == d2.selected_domain

    # Confidence must be deterministically comparable
    if d1.confidence is None or d2.confidence is None:
        assert d1.confidence == d2.confidence
    else:
        assert round(d1.confidence, 3) == round(d2.confidence, 3)

    # Status must be stable
    assert d1.status == d2.status
