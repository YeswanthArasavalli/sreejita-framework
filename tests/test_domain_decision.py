from sreejita.domains.router import decide_domain


def test_domain_decision_is_explainable(sample_df):
    decision = decide_domain(sample_df)

    # ---- structural guarantees ----
    assert decision.decision_type == "domain_detection"
    assert decision.timestamp is not None

    # ---- ambiguity-safe domain ----
    assert (
        decision.selected_domain is None
        or isinstance(decision.selected_domain, str)
    )

    # ---- confidence is non-negative if present ----
    assert decision.confidence is None or decision.confidence >= 0

    # ---- explainability contracts ----
    assert isinstance(decision.rules_applied, list)
    assert isinstance(decision.alternatives, list)
    assert isinstance(decision.signals, dict)
