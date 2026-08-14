from __future__ import annotations

from typing import Any

SEVERITY_VALUE = {
    "benign": 0.0, "informational": 0.1, "low": 0.25, "medium": 0.50,
    "high": 0.75, "critical": 1.0,
}
ATTACK_SEVERITY = {
    "benign": 0.0, "reconnaissance": 0.45, "scanning": 0.45, "brute_force": 0.60,
    "dos": 0.75, "ddos": 0.85, "web_attack": 0.80, "injection": 0.85,
    "backdoor": 0.95, "ransomware": 1.0, "exploits": 0.85, "attack": 0.65,
}
ASSET_CRITICALITY = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}


def risk_score(
    *,
    confidence: float,
    anomaly: float,
    attack_category: str,
    asset_criticality: str,
    repeated_event_count: int,
    ioc_reputation: float,
    model_agreement: float,
    uncertainty: float,
    temporal_correlation: float,
    weights: dict[str, float],
) -> float:
    values = {
        "confidence": _unit(confidence),
        "anomaly": _unit(anomaly),
        "attack_severity": ATTACK_SEVERITY.get(attack_category.lower(), 0.55),
        "asset_criticality": ASSET_CRITICALITY.get(asset_criticality.lower(), 0.5),
        "repetition": min(max(repeated_event_count, 0) / 10.0, 1.0),
        "ioc_reputation": _unit(ioc_reputation),
        "model_agreement": _unit(model_agreement),
        "uncertainty_penalty": 1.0 - _unit(uncertainty),
        "temporal_correlation": _unit(temporal_correlation),
    }
    total_weight = sum(max(float(value), 0.0) for value in weights.values()) or 1.0
    score = sum(values.get(name, 0.0) * max(float(weight), 0.0) for name, weight in weights.items()) / total_weight
    return round(min(max(score * 100, 0.0), 100.0), 2)


def severity_band(score: float) -> str:
    if score < 20:
        return "informational"
    if score < 40:
        return "low"
    if score < 60:
        return "medium"
    if score < 80:
        return "high"
    return "critical"


def _unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
