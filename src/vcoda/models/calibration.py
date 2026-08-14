from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Platt-style calibration fitted only on the validation partition."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.class_count = 0

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        probs = np.asarray(probabilities)
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(clipped)
        self.class_count = probs.shape[1]
        self.model = LogisticRegression(max_iter=2000, solver="lbfgs")
        self.model.fit(logits, y)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted")
        probs = np.asarray(probabilities)
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7))
        raw_calibrated = self.model.predict_proba(logits)
        if raw_calibrated.shape[1] == self.class_count:
            return raw_calibrated
        result = probs.copy()
        for col_idx, class_label in enumerate(self.model.classes_):
            if 0 <= class_label < self.class_count:
                result[:, class_label] = raw_calibrated[:, col_idx]
        sums = result.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return result / sums
