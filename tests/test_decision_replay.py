from sreejita.domains.router import decide_domain


def test_domain_decision_is_deterministic(sample_df):
    d1 = decide_domain(sample_df)
    d2 = decide_domain(sample_df)

    # ------------------------------------
    # Structural determinism
    # ------------------------------------
    assert d1.decision_type == d2.decision_type
    assert d1.status == d2.status

    # ------------------------------------
    # Domain outcome determinism
    # (may be None or str — both allowed)
    # ------------------------------------
    assert d1.selected_domain == d2.selected_domain

    # ------------------------------------
    # Confidence stability (NOT exact float match)
    # ------------------------------------
    if d1.confidence is not None and d2.confidence is not None:
        assert abs(d1.confidence - d2.confidence) <= 0.01

    # ------------------------------------
    # Dataset fingerprint must be identical
    # ------------------------------------
    assert d1.fingerprint == d2.fingerprint
