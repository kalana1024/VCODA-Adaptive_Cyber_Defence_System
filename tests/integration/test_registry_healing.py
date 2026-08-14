import json
from pathlib import Path

from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.self_healing import SelfHealingManager


def test_self_healing_rolls_back_corrupt_active_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("models/registry").mkdir(parents=True)
    Path("configs").mkdir()
    Path("artifacts/audit").mkdir(parents=True)
    Path("artifacts/heartbeats").mkdir(parents=True)
    Path("configs/response_policy.yaml").write_text("""mode: recommendation_only\nwindows_firewall: {allow_public_ips_only: true, rule_prefix: TEST}\nsafety: {prohibited_actions: [], protected_assets: [], protected_networks: [], allowlisted_ips: [], minimum_confidence_for_disruptive_action: 0.9, minimum_risk_for_disruptive_action: 80, require_two_signal_corroboration: true}\nactions: {informational: [continue_monitoring]}\n""")
    Path("configs/self_healing.yaml").write_text("""self_healing:\n  rollback_corrupt_models: true\n  quarantine_corrupt_artifacts: true\n  verify_audit_chain: true\n  rollback_failed_responses: false\n  stale_heartbeat_seconds: 180\n  safe_online_updates: {anomaly_model: true}\n  health_report: artifacts/reports/health.json\n""")
    old = Path("old.joblib"); old.write_bytes(b"old healthy")
    new = Path("new.joblib"); new.write_bytes(b"new healthy")
    registry = ModelRegistry()
    old_record = registry.register(model_name="m", version="1", task="supervised_binary", artifact_path=old, metrics={}, dataset_fingerprint="d", feature_list=[], preprocessing_version="1", hyperparameters={}, threshold=0.5)
    registry.promote("m", "1")
    registry.register(model_name="m", version="2", task="supervised_binary", artifact_path=new, metrics={}, dataset_fingerprint="d", feature_list=[], preprocessing_version="1", hyperparameters={}, threshold=0.5)
    registry.promote("m", "2")
    active = registry.active("supervised_binary")
    Path(active["artifact_path"]).write_bytes(b"corrupt")
    report = SelfHealingManager("configs/self_healing.yaml").run_once()
    assert any(item.get("action") == "rolled_back" for item in report["models"])
    assert registry.active("supervised_binary")["version"] == "1"
