from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from vcoda.models.domain_adaptation import split_conformal_threshold
from vcoda.models.metrics import choose_binary_threshold, evaluate_classification
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.experiments import ExperimentTracker
from vcoda.utils.io import dump_json, load_yaml, resolve_path, sha256_file

logger = logging.getLogger(__name__)


def _probability_column(frame: pd.DataFrame) -> str:
    preferred = ["probability_attack", "anomaly_confidence"]
    for column in preferred:
        if column in frame.columns:
            return column
    probability_columns = [column for column in frame.columns if column.startswith("probability_")]
    if len(probability_columns) == 1:
        candidate = probability_columns[0]
        if candidate.lower().endswith("benign"):
            raise ValueError("Prediction file does not contain a binary attack probability")
        return candidate
    if len(probability_columns) == 2:
        for column in probability_columns:
            lower = column.lower()
            if lower.endswith("attack") or lower.endswith("anomaly") or lower.endswith("_1"):
                return column
        for column in probability_columns:
            if not column.lower().endswith("benign"):
                return column
        return probability_columns[0]
    raise ValueError("Prediction file does not contain a binary attack probability")


def _model_key(path: Path) -> str:
    return path.name.replace("_validation_predictions.parquet", "").replace("_test_predictions.parquet", "")


def _drop_nan_rows(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    mask = frame[columns].notna().all(axis=1)
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(
            "Dropping %d/%d %s rows containing NaN model probabilities", dropped, len(frame), label
        )
    cleaned = frame.loc[mask].reset_index(drop=True)
    if cleaned.empty:
        raise RuntimeError(f"All {label} rows contained NaN model probabilities")
    return cleaned


def _sanitize_matrix(frame: pd.DataFrame, model_names: list[str], label: str) -> tuple[pd.DataFrame, list[str]]:
    usable_names = []
    for name in model_names:
        nan_fraction = float(frame[name].isna().mean())
        if nan_fraction >= 1.0:
            logger.warning(
                "Excluding ensemble member %r from %s: it produced no valid (non-NaN) probabilities", name, label
            )
        else:
            usable_names.append(name)
    if len(usable_names) < 2:
        raise RuntimeError(
            f"At least two {label} prediction files with valid probabilities are required, "
            f"but only {len(usable_names)} remained after excluding broken models"
        )
    frame = frame[["row_id", "actual", *usable_names]]
    return _drop_nan_rows(frame, usable_names, label), usable_names


def _load_matrix(pattern: str, suffix: str) -> tuple[pd.DataFrame, list[str]]:
    paths = [Path(path) for path in glob.glob(str(resolve_path(pattern)))]
    if not paths:
        raise FileNotFoundError(f"No prediction files matched {pattern}")
    merged: pd.DataFrame | None = None
    model_names: list[str] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "row_id" not in frame or "actual" not in frame:
            continue
        name = _model_key(path)
        probability = _probability_column(frame)
        part = frame[["row_id", "actual", probability]].rename(columns={probability: name})
        merged = part if merged is None else merged.merge(part, on=["row_id", "actual"], how="inner")
        model_names.append(name)
    if merged is None or len(model_names) < 2:
        raise RuntimeError("At least two aligned validation prediction files are required")
    return _sanitize_matrix(merged, model_names, suffix)


@dataclass
class EnsembleBundle:
    method: str
    model_names: list[str]
    weights: list[float] | None
    meta_model: Any | None
    threshold: float
    uncertainty_threshold: float
    conformal_threshold: float | None = None
    conformal_coverage: float | None = None

    def combine(self, probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        matrix = np.column_stack([np.asarray(probabilities[name], dtype=float) for name in self.model_names])
        if self.method == "stacking":
            final = self.meta_model.predict_proba(matrix)[:, 1]
        else:
            final = matrix @ np.asarray(self.weights)
        disagreement = matrix.std(axis=1)
        uncertainty = np.clip(disagreement / max(self.uncertainty_threshold, 1e-9), 0, 1)
        result = {"probability": final, "disagreement": disagreement, "uncertainty": uncertainty}
        if self.conformal_threshold is not None:
            # Split-conformal (Vovk et al.): a prediction is "confident" only if the
            # probability mass on the predicted class clears the calibrated coverage
            # bound, giving a distribution-free guarantee instead of a heuristic cutoff.
            predicted_class_nonconformity = 1 - np.maximum(final, 1 - final)
            result["conformal_confident"] = predicted_class_nonconformity <= self.conformal_threshold
        return result

    def save(self, path: str | Path) -> dict[str, str]:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return {"path": str(target), "sha256": sha256_file(target)}

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "EnsembleBundle":
        target = resolve_path(path)
        if expected_sha256 and sha256_file(target) != expected_sha256:
            raise ValueError("Ensemble checksum mismatch")
        value = joblib.load(target)
        if not isinstance(value, cls):
            raise TypeError("Invalid ensemble bundle")
        return value


def _search_weights(matrix: np.ndarray, labels: np.ndarray, iterations: int, seed: int = 42) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_weights = np.full(matrix.shape[1], 1 / matrix.shape[1])
    best_score = -1.0
    best_threshold = 0.5
    candidates = [best_weights, *rng.dirichlet(np.ones(matrix.shape[1]), size=iterations)]
    for weights in candidates:
        probability = matrix @ weights
        threshold = choose_binary_threshold(labels, probability)
        score = f1_score(labels, probability >= threshold, average="macro", zero_division=0)
        if score > best_score:
            best_weights, best_score, best_threshold = np.asarray(weights), float(score), float(threshold)
    return best_weights, best_threshold


def optimise_ensemble(config_path: str | Path = "configs/training.yaml", promote: bool = True) -> dict[str, Any]:
    config = load_yaml(config_path)
    cfg = config["ensemble"]
    validation, model_names = _load_matrix("artifacts/reports/*_binary_validation_predictions.parquet", "validation")

    conformal_cfg = cfg.get("conformal", {}) or {}
    conformal_threshold: float | None = None
    conformal_coverage: float | None = None
    fit_frame, calibration_frame = validation, None
    if bool(conformal_cfg.get("enabled", True)) and len(validation) >= 20:
        from sklearn.model_selection import train_test_split
        fit_frame, calibration_frame = train_test_split(
            validation, test_size=float(conformal_cfg.get("calibration_fraction", 0.3)),
            stratify=validation["actual"], random_state=42,
        )
        fit_frame = fit_frame.reset_index(drop=True)
        calibration_frame = calibration_frame.reset_index(drop=True)

    x_validation = fit_frame[model_names].to_numpy(dtype=float)
    y_validation = fit_frame["actual"].to_numpy(dtype=int)
    method = str(cfg.get("method", "stacking"))
    if method == "stacking":
        meta_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        meta_model.fit(x_validation, y_validation)
        validation_probability = meta_model.predict_proba(x_validation)[:, 1]
        weights = None
        threshold = choose_binary_threshold(y_validation, validation_probability)
    elif method == "weighted_soft_voting":
        weights_array, threshold = _search_weights(x_validation, y_validation, int(cfg.get("weight_search_iterations", 3000)))
        weights = weights_array.tolist()
        meta_model = None
    else:
        raise ValueError("ensemble.method must be stacking or weighted_soft_voting")

    if calibration_frame is not None:
        # Nonconformity computed on a held-out slice never used to fit the meta-model
        # or weights, so the resulting quantile is a valid split-conformal calibration.
        x_calibration = calibration_frame[model_names].to_numpy(dtype=float)
        y_calibration = calibration_frame["actual"].to_numpy(dtype=int)
        if method == "stacking":
            calibration_probability = meta_model.predict_proba(x_calibration)[:, 1]
        else:
            calibration_probability = x_calibration @ np.asarray(weights)
        true_class_probability = np.where(y_calibration == 1, calibration_probability, 1 - calibration_probability)
        nonconformity_scores = 1 - true_class_probability
        conformal_coverage = float(conformal_cfg.get("coverage", 0.9))
        conformal_threshold = split_conformal_threshold(nonconformity_scores, coverage=conformal_coverage)

    test_paths = { _model_key(Path(path)): Path(path) for path in glob.glob(str(resolve_path("artifacts/reports/*_test_predictions.parquet"))) }
    missing = [name for name in model_names if name not in test_paths]
    if missing:
        raise RuntimeError(f"Test predictions are missing for ensemble members: {missing}")
    test: pd.DataFrame | None = None
    for name in model_names:
        frame = pd.read_parquet(test_paths[name])
        probability = _probability_column(frame)
        part = frame[["row_id", "actual", probability]].rename(columns={probability: name})
        test = part if test is None else test.merge(part, on=["row_id", "actual"], how="inner")
    if test is None or test.empty:
        raise RuntimeError("No aligned test predictions were available")
    test = _drop_nan_rows(test, model_names, "test")
    bundle = EnsembleBundle(
        method=method, model_names=model_names, weights=weights, meta_model=meta_model,
        threshold=threshold, uncertainty_threshold=float(cfg.get("uncertainty_threshold", 0.25)),
        conformal_threshold=conformal_threshold, conformal_coverage=conformal_coverage,
    )
    combined = bundle.combine({name: test[name].to_numpy() for name in model_names})
    probabilities = np.column_stack([1 - combined["probability"], combined["probability"]])
    metrics = evaluate_classification(test["actual"].to_numpy(), probabilities, classes=["benign", "attack"], threshold=threshold)
    metrics["mean_disagreement"] = float(combined["disagreement"].mean())
    if "conformal_confident" in combined:
        metrics["conformal_coverage_target"] = conformal_coverage
        metrics["conformal_threshold"] = conformal_threshold
        metrics["conformal_confident_fraction"] = float(np.mean(combined["conformal_confident"]))
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = resolve_path(f"models/ensemble/{method}_{version}.joblib")
    bundle.save(staging)
    registry = ModelRegistry()
    active_binary = registry.active("supervised_binary") or {}
    record = registry.register(
        model_name=f"ensemble_{method}", version=version, task="ensemble_binary", artifact_path=staging,
        metrics=metrics, dataset_fingerprint=active_binary.get("dataset_fingerprint", "unknown"),
        feature_list=model_names, preprocessing_version="ensemble-1.0.0",
        hyperparameters={"method": method, "weights": weights, "members": model_names}, threshold=threshold,
    )
    if promote:
        registry.promote(record["model_name"], record["version"])
    report = {"method": method, "members": model_names, "weights": weights, "threshold": threshold, "metrics": metrics, "registry_record": record}
    report_path = "artifacts/reports/ensemble_report.json"
    dump_json(report, report_path)
    tracker = ExperimentTracker("ensemble_binary", backend="local")
    tracker.log_parameters({"method": method, "members": model_names, "weights": weights})
    tracker.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
    tracker.log_artifact(report_path)
    tracker.log_artifact(staging)
    tracker.finish()
    return report
