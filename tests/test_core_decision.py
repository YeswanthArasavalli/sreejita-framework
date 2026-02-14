from sreejita.core.decision import DecisionExplanation


def test_confidence_without_evidence_is_downgraded_to_zero_and_recorded():
    d = DecisionExplanation(selected_domain="healthcare", confidence=0.8, status="detected")
    assert d.confidence == 0.0
    assert d.status == "insufficient_data"
    assert "downgrade_no_evidence_for_confidence" in d.rules_applied


def test_non_positive_signal_strength_forces_insufficient_data():
    d = DecisionExplanation(
        selected_domain="healthcare",
        confidence=0.9,
        status="detected",
        signals={"signal_strength": 0.0},
    )
    assert d.status == "insufficient_data"
    assert "downgrade_non_positive_signal_strength" in d.rules_applied


def test_detected_requires_high_confidence():
    d = DecisionExplanation(
        selected_domain="healthcare",
        confidence=0.74,
        status="detected",
        signals={"signal_strength": 0.9},
    )
    assert d.status == "ambiguous"
    assert "downgrade_detected_requires_high_confidence" in d.rules_applied


def test_ambiguous_downgrades_under_very_low_confidence():
    d = DecisionExplanation(
        selected_domain="healthcare",
        confidence=0.2,
        status="ambiguous",
        signals={"signal_strength": 0.9},
    )
    assert d.status == "insufficient_data"
    assert "downgrade_ambiguous_under_low_confidence" in d.rules_applied


def test_api_shape_unchanged_for_serialization():
    d = DecisionExplanation(
        selected_domain="healthcare",
        confidence=0.8,
        status="detected",
        signals={"signal_strength": 0.9},
    )
    payload = d.to_dict()
    assert set(payload) == {
        "decision_id",
        "decision_type",
        "selected_domain",
        "confidence",
        "status",
        "alternatives",
        "signals",
        "rules_applied",
        "domain_scores",
        "fingerprint",
        "timestamp",
    }
