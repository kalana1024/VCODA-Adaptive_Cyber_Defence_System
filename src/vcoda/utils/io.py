from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    return Path(os.getenv("VCODA_PROJECT_ROOT", Path.cwd())).resolve()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def ensure_within(base: str | Path, candidate: str | Path) -> Path:
    base_path = resolve_path(base)
    candidate_path = resolve_path(candidate)
    try:
        candidate_path.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Unsafe path outside allowed directory: {candidate_path}") from exc
    return candidate_path


def safe_filename(name: str) -> str:
    clean = Path(name).name
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", clean)
    if not clean or clean in {".", ".."}:
        raise ValueError("Invalid file name")
    return clean


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))


def load_json(path: str | Path, default: Any = None) -> Any:
    target = resolve_path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, sort_keys=True, default=str))


def atomic_write_text(path: str | Path, text: str) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent) as tmp:
        tmp.write(text)
        temp_name = tmp.name
    os.replace(temp_name, target)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    target = resolve_path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number} in {target}") from exc


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
