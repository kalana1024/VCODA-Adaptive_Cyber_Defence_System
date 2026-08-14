from __future__ import annotations

import importlib.util
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcoda.utils.io import append_jsonl, dump_json


class ExperimentTracker:
    def __init__(self, experiment: str, backend: str = "local") -> None:
        self.experiment = experiment
        self.backend = backend
        self.run_id = str(uuid.uuid4())
        self.record: dict[str, Any] = {
            "experiment": experiment,
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "parameters": {},
            "metrics": {},
            "artifacts": [],
        }
        self.mlflow = None
        if backend == "mlflow" and importlib.util.find_spec("mlflow") is not None:
            import mlflow
            if os.getenv("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow

    def log_parameters(self, values: dict[str, Any]) -> None:
        self.record["parameters"].update(values)
        if self.mlflow:
            self.mlflow.log_params({key: str(value)[:500] for key, value in values.items()})

    def log_metrics(self, values: dict[str, float]) -> None:
        numeric = {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}
        self.record["metrics"].update(numeric)
        if self.mlflow:
            self.mlflow.log_metrics(numeric)

    def log_artifact(self, path: str | Path) -> None:
        self.record["artifacts"].append(str(path))
        if self.mlflow:
            self.mlflow.log_artifact(str(path))

    def finish(self, status: str = "finished") -> dict[str, Any]:
        self.record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.record["status"] = status
        append_jsonl("artifacts/reports/experiment_runs.jsonl", self.record)
        dump_json(self.record, f"artifacts/reports/experiment_{self.run_id}.json")
        return self.record

    def __enter__(self) -> "ExperimentTracker":
        if self.mlflow:
            self.mlflow.start_run(run_name=self.run_id)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.finish("failed" if exc else "finished")
        if self.mlflow:
            self.mlflow.end_run(status="FAILED" if exc else "FINISHED")
