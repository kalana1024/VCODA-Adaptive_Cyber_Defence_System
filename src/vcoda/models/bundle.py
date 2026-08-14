from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from vcoda.models.calibration import ProbabilityCalibrator
from vcoda.models.preprocessing import VCODAPreprocessor
from vcoda.utils.io import sha256_file


@dataclass
class ModelBundle:
    name: str
    task: str
    model: Any
    preprocessor: VCODAPreprocessor
    classes: list[str]
    threshold: float | None = None
    calibrator: ProbabilityCalibrator | None = None
    metadata: dict[str, Any] | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame)
        probabilities = np.asarray(self.model.predict_proba(transformed))
        if self.calibrator is not None:
            probabilities = self.calibrator.transform(probabilities)
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(frame)
        if self.task == "binary":
            threshold = 0.5 if self.threshold is None else self.threshold
            return (probabilities[:, 1] >= threshold).astype(int)
        return probabilities.argmax(axis=1)

    def schema_report(self, frame: pd.DataFrame) -> dict[str, Any]:
        _, report = self.preprocessor.align(frame)
        return report

    def save(self, path: str | Path) -> dict[str, str]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return {"path": str(target), "sha256": sha256_file(target)}

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "ModelBundle":
        target = Path(path)
        if expected_sha256 and sha256_file(target) != expected_sha256:
            raise ValueError(f"Model checksum mismatch: {target}")
        value = joblib.load(target)
        if not isinstance(value, cls):
            raise TypeError(f"Untrusted or incompatible model bundle: {target}")
        return value
