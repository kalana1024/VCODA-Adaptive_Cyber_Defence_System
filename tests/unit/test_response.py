from pathlib import Path

from vcoda.response.engine import ResponseEngine

POLICY = """mode: recommendation_only
windows_firewall:
  enabled: false
  allow_public_ips_only: true
  rule_prefix: TEST
safety:
  prohibited_actions: [arbitrary_command]
  protected_assets: [database_primary]
  protected_networks: [127.0.0.0/8]
  allowlisted_ips: [127.0.0.1]
  minimum_confidence_for_disruptive_action: 0.9
  minimum_risk_for_disruptive_action: 80
  require_two_signal_corroboration: true
  rollback_required: true
actions:
  critical: [temporary_block_source]
  informational: [continue_monitoring]
"""


def test_response_defaults_to_recommendation_and_protects_assets(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY)
    engine = ResponseEngine(path)
    decision = engine.decide(
        action="temporary_block_source", source_ip="8.8.8.8", confidence=0.99,
        risk_score=95, corroborating_signals=3, asset_type="database_primary",
    )
    assert decision.allowed is False
    assert any("Protected asset" in reason for reason in decision.reasons)


def test_expired_response_is_rolled_back_only_once(tmp_path):
    from datetime import datetime, timedelta, timezone

    from vcoda.utils.io import append_jsonl

    policy = POLICY.replace("mode: recommendation_only", "mode: simulation")
    path = tmp_path / "simulation_policy.yaml"
    path.write_text(policy)
    engine = ResponseEngine(path)
    engine.ledger = tmp_path / "response_ledger.jsonl"
    append_jsonl(
        engine.ledger,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "temporary_block_source",
            "mode": "simulation",
            "allowed": True,
            "requires_confirmation": False,
            "executed": True,
            "rollback_id": "rollback-once",
            "reasons": [],
            "source_ip": "8.8.8.8",
            "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        },
    )
    first = engine.rollback_expired()
    second = engine.rollback_expired()
    assert first == [{"rolled_back": True, "rollback_id": "rollback-once", "mode": "simulation"}]
    assert second == []
