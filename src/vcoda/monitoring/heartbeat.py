from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcoda.utils.io import dump_json, load_json


def write_heartbeat(service: str, details: dict[str, Any] | None = None) -> Path:
    path = Path("artifacts/heartbeats") / f"{service}.json"
    dump_json({
        "service": service,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }, path)
    return path


def read_heartbeat(service: str) -> dict[str, Any] | None:
    return load_json(Path("artifacts/heartbeats") / f"{service}.json")
