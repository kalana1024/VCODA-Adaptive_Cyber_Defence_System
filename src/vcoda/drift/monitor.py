from __future__ import annotations

import importlib.util
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vcoda.utils.io import append_jsonl, dump_json


class _FallbackPageHinkley:
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.999) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.count = 0
        self.drift_detected = False

    def update(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative = self.alpha * self.cumulative + value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        self.drift_detected = (self.cumulative - self.minimum) > self.threshold
        if self.drift_detected:
            self.cumulative = 0.0
            self.minimum = 0.0
            self.count = 0


class DriftMonitor:
    def __init__(self, window_size: int = 2000, output_dir: str | Path = "artifacts/drift") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self.detectors: dict[str, Any] = {}
        self.backend = "fallback_page_hinkley"
        if importlib.util.find_spec("river") is not None:
            from river import drift
            self.detectors = {
                "confidence": drift.ADWIN(delta=0.002),
                "anomaly": drift.PageHinkley(min_instances=50, delta=0.005, threshold=30),
                "error": drift.ADWIN(delta=0.002),
                "attack_rate": drift.ADWIN(delta=0.002),
            }
            self.backend = "river"
        else:
            self.detectors = {name: _FallbackPageHinkley() for name in ["confidence", "anomaly", "error", "attack_rate"]}
        self.events: list[dict[str, Any]] = []

    def update(
        self,
        *,
        event_id: str,
        confidence: float,
        anomaly_score: float,
        predicted_attack: bool,
        error: int | None = None,
        features: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        record = {
            "event_id": event_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": float(confidence), "anomaly_score": float(anomaly_score),
            "predicted_attack": bool(predicted_attack), "error": error, "features": features or {},
        }
        self.window.append(record)
        values = {"confidence": confidence, "anomaly": anomaly_score, "attack_rate": float(predicted_attack)}
        if error is not None:
            values["error"] = float(error)
        detected: list[dict[str, Any]] = []
        for name, value in values.items():
            detector = self.detectors[name]
            detector.update(float(value))
            if bool(getattr(detector, "drift_detected", False)):
                event = self._record_drift(name, float(value))
                detected.append(event)
        return detected

    def _record_drift(self, signal: str, current_value: float) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_stamp = timestamp.replace(":", "-")
        event = {
            "timestamp": timestamp,
            "signal": signal,
            "current_value": current_value,
            "backend": self.backend,
            "action": "saved_window_and_recommend_retraining",
            "automatic_supervised_retraining": False,
            "window_size": len(self.window),
        }
        window_path = self.output_dir / f"drift_window_{signal}_{safe_stamp}.json"
        dump_json(list(self.window), window_path)
        event["window_path"] = str(window_path)
        append_jsonl(self.output_dir / "drift_events.jsonl", event)
        self.events.append(event)
        return event

    def status(self) -> dict[str, Any]:
        return {"backend": self.backend, "events": len(self.events), "last_event": self.events[-1] if self.events else None, "window_rows": len(self.window)}
