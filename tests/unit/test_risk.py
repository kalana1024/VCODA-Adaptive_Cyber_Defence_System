from vcoda.risk import risk_score, severity_band


def test_risk_is_bounded_and_severity_is_stable():
    score = risk_score(
        confidence=1, anomaly=1, attack_category="ransomware", asset_criticality="critical",
        repeated_event_count=50, ioc_reputation=1, model_agreement=1, uncertainty=0,
        temporal_correlation=1,
        weights={"confidence": 1, "anomaly": 1, "attack_severity": 1, "asset_criticality": 1,
                 "repetition": 1, "ioc_reputation": 1, "model_agreement": 1,
                 "uncertainty_penalty": 1, "temporal_correlation": 1},
    )
    assert 0 <= score <= 100
    assert severity_band(score) == "critical"
