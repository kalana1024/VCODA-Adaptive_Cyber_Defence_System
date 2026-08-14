from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcoda.utils.io import append_jsonl, iter_jsonl, sha256_json

GENESIS_HASH = "0" * 64


class AuditChain:
    def __init__(self, path: str | Path = "artifacts/audit/audit.jsonl") -> None:
        self.path = Path(path)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        previous_hash = GENESIS_HASH
        last_sequence = 0
        for record in iter_jsonl(self.path) or []:
            previous_hash = record["record_hash"]
            last_sequence = int(record["sequence"])
        body = {
            "sequence": last_sequence + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_hash": previous_hash,
            "event": event,
        }
        body["record_hash"] = sha256_json(body)
        append_jsonl(self.path, body)
        return body

    def verify(self) -> dict[str, Any]:
        previous_hash = GENESIS_HASH
        count = 0
        for count, record in enumerate(iter_jsonl(self.path) or [], 1):
            received = record.get("record_hash")
            body = {key: value for key, value in record.items() if key != "record_hash"}
            expected = sha256_json(body)
            if record.get("previous_hash") != previous_hash:
                return {"valid": False, "records": count, "error": "previous_hash_mismatch"}
            if received != expected:
                return {"valid": False, "records": count, "error": "record_hash_mismatch"}
            previous_hash = str(received)
        return {"valid": True, "records": count, "last_hash": previous_hash}
