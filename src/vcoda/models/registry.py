from __future__ import annotations

import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcoda.utils.io import dump_json, load_json, resolve_path, sha256_file


class ModelRegistry:
    def __init__(self, root: str | Path = "models/registry") -> None:
        self.root = resolve_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.active_path = self.root / "active.json"

    def register(
        self,
        *,
        model_name: str,
        version: str,
        task: str,
        artifact_path: str | Path,
        metrics: dict[str, Any],
        dataset_fingerprint: str,
        feature_list: list[str],
        preprocessing_version: str,
        hyperparameters: dict[str, Any],
        threshold: float | None,
        git_commit: str | None = None,
    ) -> dict[str, Any]:
        source = resolve_path(artifact_path)
        if not source.exists():
            raise FileNotFoundError(source)
        destination_dir = self.root / model_name / version
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        record = {
            "model_name": model_name,
            "version": version,
            "task": task,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_path": str(destination.relative_to(resolve_path("."))),
            "artifact_sha256": sha256_file(destination),
            "dataset_fingerprint": dataset_fingerprint,
            "feature_list": feature_list,
            "preprocessing_version": preprocessing_version,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "threshold": threshold,
            "git_commit": git_commit,
            "environment": {"python": sys.version, "platform": platform.platform()},
            "status": "registered",
        }
        dump_json(record, destination_dir / "metadata.json")
        index = load_json(self.index_path, default={"models": []})
        index["models"].append(record)
        dump_json(index, self.index_path)
        return record

    def list(self, task: str | None = None) -> list[dict[str, Any]]:
        models = load_json(self.index_path, default={"models": []})["models"]
        return [model for model in models if task is None or model["task"] == task]

    def promote(self, model_name: str, version: str) -> dict[str, Any]:
        matches = [m for m in self.list() if m["model_name"] == model_name and m["version"] == version]
        if not matches:
            raise KeyError(f"Model not found: {model_name}:{version}")
        record = matches[-1]
        path = resolve_path(record["artifact_path"])
        if sha256_file(path) != record["artifact_sha256"]:
            raise ValueError("Cannot promote a model whose checksum is invalid")
        active = load_json(self.active_path, default={"history": [], "active": {}})
        task = record["task"]
        if task in active["active"]:
            active["history"].append(active["active"][task])
        active["active"][task] = record
        dump_json(active, self.active_path)
        return record

    def active(self, task: str) -> dict[str, Any] | None:
        return load_json(self.active_path, default={"active": {}}).get("active", {}).get(task)

    def latest(self, model_name: str) -> dict[str, Any] | None:
        """Most recently registered record for a specific model_name, regardless of
        whether it is the "active" promotion for its task. Needed because several
        architectures (mlp/cnn1d/cnn_lstm/transformer) share one "deep_binary" active
        slot, but an ensemble may need all of them loaded simultaneously."""
        matches = [record for record in self.list() if record["model_name"] == model_name]
        if not matches:
            return None
        return max(matches, key=lambda record: record["created_at"])

    def rollback(self, task: str) -> dict[str, Any]:
        state = load_json(self.active_path, default={"history": [], "active": {}})
        candidates = [record for record in reversed(state.get("history", [])) if record.get("task") == task]
        if not candidates:
            raise RuntimeError(f"No rollback candidate for task {task}")
        candidate = candidates[0]
        path = resolve_path(candidate["artifact_path"])
        if not path.exists() or sha256_file(path) != candidate["artifact_sha256"]:
            raise RuntimeError("Rollback candidate is missing or corrupt")
        current = state.get("active", {}).get(task)
        if current:
            state["history"].append(current)
        state["active"][task] = candidate
        dump_json(state, self.active_path)
        return candidate
