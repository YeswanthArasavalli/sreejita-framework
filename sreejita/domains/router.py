def decide_domain(df: pd.DataFrame) -> DecisionExplanation:
    """
    Stabilization-mode domain decision.

    GUARANTEES:
    - No domain is forced
    - Ambiguity is explicit
    - Generic is NOT auto-selected
    - Downstream execution can be safely gated
    """

    rule_results: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------
    # PHASE 1 — RULE-BASED DETECTION (READ-ONLY)
    # -------------------------------------------------
    for detector in DOMAIN_DETECTORS:
        try:
            result = detector.detect(df)
            if not result or not getattr(result, "domain", None):
                continue

            prev = rule_results.get(result.domain)
            if prev is None or result.confidence > prev["confidence"]:
                rule_results[result.domain] = {
                    "confidence": float(result.confidence or 0.0),
                    "signals": result.signals or {},
                    "detector": detector.__class__.__name__,
                }
        except Exception:
            continue

    # -------------------------------------------------
    # PHASE 2 — SELECT BEST DOMAIN (NO FORCING)
    # -------------------------------------------------
    selected_domain: Optional[str] = None
    confidence: float = 0.0
    status: str = "insufficient_data"
    meta: Dict[str, Any] = {}

    if rule_results:
        selected_domain, meta = max(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
        )
        confidence = float(meta.get("confidence", 0.0))

        if confidence >= MIN_DOMAIN_CONFIDENCE:
            status = "detected"
        else:
            selected_domain = None
            status = "ambiguous"

    # -------------------------------------------------
    # EXPLAINABILITY — ALTERNATIVES
    # -------------------------------------------------
    alternatives = [
        {
            "domain": d,
            "confidence": round(info["confidence"], 2),
        }
        for d, info in sorted(
            rule_results.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True,
        )
    ]

    # -------------------------------------------------
    # DECISION OBJECT (CANONICAL, HONEST)
    # -------------------------------------------------
    decision = DecisionExplanation(
        decision_type="domain_detection",
        selected_domain=selected_domain,
        confidence=round(confidence, 2),
        status=status,
        alternatives=alternatives,
        signals=meta.get("signals", {}),
        rules_applied=[
            "rule_based_detection",
            "highest_confidence_wins",
            "ambiguity_allowed",
        ],
        domain_scores={
            d: {"confidence": v["confidence"]}
            for d, v in rule_results.items()
        },
    )

    # -------------------------------------------------
    # ENGINE ATTACHMENT (ONLY IF SAFE)
    # -------------------------------------------------
    if status == "detected" and selected_domain in _DOMAIN_FACTORY:
        decision.attach_engine(_get_domain_engine(selected_domain))

    # -------------------------------------------------
    # TRACEABILITY
    # -------------------------------------------------
    decision.fingerprint = dataframe_fingerprint(df)

    # -------------------------------------------------
    # OBSERVABILITY
    # -------------------------------------------------
    for observer in _OBSERVERS:
        try:
            observer.record(decision)
        except Exception:
            pass

    return decision
