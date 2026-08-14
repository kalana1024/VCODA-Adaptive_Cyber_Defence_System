from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcoda.audit.chain import AuditChain
from vcoda.models.registry import ModelRegistry
from vcoda.response.engine import ResponseEngine
from vcoda.utils.io import dump_json, load_json, load_yaml, resolve_path, sha256_file


class SelfHealingManager:
    """Controlled recovery: checksum validation, rollback, quarantine, heartbeat and reversible response cleanup."""

    def __init__(self, config_path: str | Path = "configs/self_healing.yaml") -> None:
        self.config = load_yaml(config_path)["self_healing"]
        self.registry = ModelRegistry()
        self.audit = AuditChain()
        self.response = ResponseEngine()
        self.quarantine = resolve_path("artifacts/quarantine")
        self.quarantine.mkdir(parents=True, exist_ok=True)

    def check_models(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        state = load_json(self.registry.active_path, default={"active": {}})
        for task, record in state.get("active", {}).items():
            path = resolve_path(record["artifact_path"])
            healthy = path.exists() and sha256_file(path) == record.get("artifact_sha256")
            result = {"component": f"model:{task}", "healthy": healthy, "path": str(path), "action": None}
            if not healthy and self.config.get("rollback_corrupt_models", True):
                if path.exists() and self.config.get("quarantine_corrupt_artifacts", True):
                    destination = self.quarantine / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{path.name}"
                    shutil.move(path, destination)
                    result["quarantined_to"] = str(destination)
                try:
                    replacement = self.registry.rollback(task)
                    result["action"] = "rolled_back"
                    result["replacement"] = replacement
                except Exception as exc:
                    result["action"] = "manual_recovery_required"
                    result["error"] = str(exc)
            results.append(result)
        return results

    def check_heartbeats(self) -> list[dict[str, Any]]:
        threshold = float(self.config.get("stale_heartbeat_seconds", 180))
        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []
        for path in resolve_path("artifacts/heartbeats").glob("*.json"):
            heartbeat = load_json(path, default={})
            try:
                timestamp = datetime.fromisoformat(heartbeat["timestamp"])
                age = (now - timestamp).total_seconds()
            except Exception:
                age = float("inf")
            results.append({"component": f"service:{path.stem}", "healthy": age <= threshold, "age_seconds": age, "automatic_restart": False})
        return results

    def run_once(self) -> dict[str, Any]:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": self.check_models(),
            "audit": self.audit.verify() if self.config.get("verify_audit_chain", True) else {"skipped": True},
            "heartbeats": self.check_heartbeats(),
            "response_rollbacks": self.response.rollback_expired() if self.config.get("rollback_failed_responses", True) else [],
            "auto_retrain_main_models": False,
            "safe_online_updates": self.config.get("safe_online_updates", {}),
        }
        report["healthy"] = all(item.get("healthy", True) for item in report["models"] + report["heartbeats"]) and report["audit"].get("valid", True)
        dump_json(report, self.config.get("health_report", "artifacts/reports/self_healing_status.json"))
        return report

    def watch(self) -> None:
        interval = int(self.config.get("interval_seconds", 60))
        while True:
            self.run_once()
            time.sleep(interval)
