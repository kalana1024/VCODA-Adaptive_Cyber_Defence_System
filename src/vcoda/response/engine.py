from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vcoda.security import is_protected_ip, is_public_ip
from vcoda.utils.io import append_jsonl, iter_jsonl, load_yaml
from vcoda.utils.system import detect_hardware


@dataclass
class ResponseDecision:
    action: str
    mode: str
    allowed: bool
    requires_confirmation: bool
    executed: bool
    rollback_id: str | None
    reasons: list[str]


class ResponseEngine:
    def __init__(self, policy_path: str | Path = "configs/response_policy.yaml") -> None:
        self.policy = load_yaml(policy_path)
        self.ledger = Path("artifacts/audit/response_ledger.jsonl")

    def recommend(self, severity: str, attack_category: str, context: dict[str, Any]) -> str:
        candidates = self.policy.get("actions", {}).get(severity, ["manual_investigation"])
        if attack_category.lower() in {"dos", "ddos", "reconnaissance", "scanning"} and "temporary_block_source" in candidates:
            return "temporary_block_source"
        if severity in {"high", "critical"} and "isolate_endpoint" in candidates:
            return "isolate_endpoint"
        return str(candidates[0])

    def decide(
        self,
        *,
        action: str,
        source_ip: str | None,
        confidence: float,
        risk_score: float,
        corroborating_signals: int,
        asset_type: str | None,
        analyst_confirmed: bool = False,
    ) -> ResponseDecision:
        mode = str(self.policy.get("mode", "recommendation_only"))
        safety = self.policy["safety"]
        reasons: list[str] = []
        disruptive = action in {"temporary_block_source", "temporary_rate_limit", "isolate_endpoint", "disable_vulnerable_service", "force_credential_reset"}
        if action in safety.get("prohibited_actions", []):
            reasons.append("Action is prohibited by policy")
        if asset_type in safety.get("protected_assets", []):
            reasons.append("Protected asset requires explicit analyst control")
        if disruptive and confidence < float(safety.get("minimum_confidence_for_disruptive_action", 0.92)):
            reasons.append("Confidence is below the disruptive-action threshold")
        if disruptive and risk_score < float(safety.get("minimum_risk_for_disruptive_action", 85)):
            reasons.append("Risk score is below the disruptive-action threshold")
        if disruptive and safety.get("require_two_signal_corroboration", True) and corroborating_signals < 2:
            reasons.append("Fewer than two independent signals corroborate the threat")
        if source_ip and is_protected_ip(source_ip, safety.get("protected_networks", []), safety.get("allowlisted_ips", [])):
            reasons.append("Source address is protected or allowlisted")
        requires_confirmation = disruptive
        if mode == "recommendation_only":
            reasons.append("Response mode is recommendation_only")
        elif mode == "simulation":
            requires_confirmation = False
        elif mode == "human_approved" and not analyst_confirmed:
            reasons.append("Analyst confirmation is required")
        elif mode == "windows_firewall" and not analyst_confirmed:
            reasons.append("Windows Firewall changes require explicit confirmation")
        allowed = not reasons or (mode == "simulation" and not any("prohibited" in value.lower() or "protected" in value.lower() or "allowlisted" in value.lower() for value in reasons))
        return ResponseDecision(action, mode, allowed, requires_confirmation, False, None, reasons)

    def execute(self, decision: ResponseDecision, source_ip: str | None, duration_minutes: int = 5) -> ResponseDecision:
        if not decision.allowed:
            self._log(decision, source_ip)
            return decision
        if decision.mode == "recommendation_only":
            self._log(decision, source_ip)
            return decision
        rollback_id = str(uuid.uuid4())
        if decision.mode == "simulation":
            decision.executed = True
            decision.rollback_id = rollback_id
            self._log(decision, source_ip, duration_minutes)
            return decision
        if decision.mode in {"human_approved", "windows_firewall"} and decision.action == "temporary_block_source":
            if os.name != "nt":
                decision.allowed = False
                decision.reasons.append("Windows Firewall integration is available only on Windows")
            elif not source_ip:
                decision.allowed = False
                decision.reasons.append("A source IP is required")
            elif self.policy["windows_firewall"].get("allow_public_ips_only", True) and not is_public_ip(source_ip):
                decision.allowed = False
                decision.reasons.append("Only public source IPs may be blocked by this policy")
            elif not detect_hardware().get("is_admin"):
                decision.allowed = False
                decision.reasons.append("Administrator privileges are required")
            else:
                rule_name = f"{self.policy['windows_firewall'].get('rule_prefix', 'VCODA_TEMP_BLOCK')}_{rollback_id}"
                command = ["netsh", "advfirewall", "firewall", "add", "rule", f"name={rule_name}", "dir=in", "action=block", f"remoteip={source_ip}"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False, shell=False)
                if result.returncode == 0:
                    decision.executed, decision.rollback_id = True, rollback_id
                else:
                    decision.allowed = False
                    decision.reasons.append(f"Windows Firewall rejected the rule: {result.stderr.strip()[:300]}")
        self._log(decision, source_ip, duration_minutes)
        return decision

    def rollback(self, rollback_id: str) -> dict[str, Any]:
        records = list(iter_jsonl(self.ledger) or [])
        matches = [record for record in records if record.get("rollback_id") == rollback_id and record.get("executed")]
        if not matches:
            return {"rolled_back": False, "reason": "rollback record not found"}
        record = matches[-1]
        if record["mode"] == "simulation":
            result = {"rolled_back": True, "rollback_id": rollback_id, "mode": "simulation"}
        elif os.name == "nt":
            rule_name = f"{self.policy['windows_firewall'].get('rule_prefix', 'VCODA_TEMP_BLOCK')}_{rollback_id}"
            command = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False, shell=False)
            result = {"rolled_back": completed.returncode == 0, "rollback_id": rollback_id, "stderr": completed.stderr[:300]}
        else:
            result = {"rolled_back": False, "reason": "Windows Firewall rollback is unavailable on this OS"}
        append_jsonl(self.ledger, {"timestamp": datetime.now(timezone.utc).isoformat(), "event": "rollback", **result})
        return result

    def rollback_expired(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        records = list(iter_jsonl(self.ledger) or [])
        completed = {
            str(record.get("rollback_id"))
            for record in records
            if record.get("event") == "rollback" and record.get("rolled_back")
        }
        results = []
        for record in records:
            rollback_id = str(record.get("rollback_id") or "")
            if (
                record.get("executed")
                and rollback_id
                and rollback_id not in completed
                and record.get("expires_at")
            ):
                expiry = datetime.fromisoformat(record["expires_at"])
                if expiry <= now:
                    result = self.rollback(rollback_id)
                    results.append(result)
                    if result.get("rolled_back"):
                        completed.add(rollback_id)
        return results

    def _log(self, decision: ResponseDecision, source_ip: str | None, duration_minutes: int = 5) -> None:
        record = asdict(decision)
        record.update({
            "timestamp": datetime.now(timezone.utc).isoformat(), "source_ip": source_ip,
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat() if decision.executed else None,
        })
        append_jsonl(self.ledger, record)
