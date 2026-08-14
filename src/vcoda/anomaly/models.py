from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from vcoda.data.loaders import load_manifest, load_split, split_xy
from vcoda.models.metrics import evaluate_classification
from vcoda.models.preprocessing import VCODAPreprocessor
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.experiments import ExperimentTracker
from vcoda.utils.io import dump_json, load_yaml, resolve_path, sha256_file


@dataclass
class AnomalyBundle:
    name: str
    model: Any
    preprocessor: VCODAPreprocessor
    threshold: float
    score_low: float
    score_high: float
    feature_baseline: dict[str, dict[str, float]]

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.preprocessor.transform(frame)
        if hasattr(self.model, "score_samples"):
            raw = -np.asarray(self.model.score_samples(x), dtype=float)
        else:
            raw = -np.asarray(self.model.decision_function(x), dtype=float)
        return raw

    def confidence(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.scores(frame)
        return np.clip((scores - self.score_low) / max(self.score_high - self.score_low, 1e-9), 0, 1)

    def explain(self, frame: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, float]]]:
        aligned, _ = self.preprocessor.align(frame)
        explanations: list[list[dict[str, float]]] = []
        for _, row in aligned.iterrows():
            contributions: list[tuple[str, float]] = []
            for feature, stats in self.feature_baseline.items():
                value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                z = abs((float(value) - stats["median"]) / max(stats["mad"], 1e-9))
                contributions.append((feature, z))
            explanations.append([{"feature": name, "robust_deviation": float(score)} for name, score in sorted(contributions, key=lambda item: item[1], reverse=True)[:top_k]])
        return explanations

    def save(self, path: str | Path) -> dict[str, str]:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return {"path": str(target), "sha256": sha256_file(target)}

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "AnomalyBundle":
        target = resolve_path(path)
        if expected_sha256 and sha256_file(target) != expected_sha256:
            raise ValueError("Anomaly model checksum mismatch")
        value = joblib.load(target)
        if not isinstance(value, cls):
            raise TypeError("Invalid anomaly model bundle")
        return value


def _threshold_from_validation(scores: np.ndarray, labels: np.ndarray, target_benign_fpr: float) -> float:
    benign_scores = scores[labels == 0]
    if len(benign_scores) == 0:
        raise ValueError("Validation data contains no benign records")
    percentile_threshold = float(np.quantile(benign_scores, 1 - target_benign_fpr))
    if len(np.unique(labels)) < 2:
        return percentile_threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    best = percentile_threshold
    best_f1 = -1.0
    for index, threshold in enumerate(thresholds):
        predictions = (scores >= threshold).astype(int)
        benign_fpr = float(predictions[labels == 0].mean())
        if benign_fpr > target_benign_fpr:
            continue
        p, r = precision[index], recall[index]
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best, best_f1 = float(threshold), float(f1)
    return best


def train_anomaly_models(config_path: str | Path = "configs/training.yaml", models: list[str] | None = None, promote_best: bool = True) -> dict[str, Any]:
    config = load_yaml(config_path)
    cfg = config["anomaly"]
    max_rows = int(cfg.get("max_training_rows", 500_000))
    train_frame = load_split("train", config["dataset"]["prepared_dir"], max_rows=max_rows)
    validation_frame = load_split("validation", config["dataset"]["prepared_dir"], max_rows=max_rows)
    test_frame = load_split("test", config["dataset"]["prepared_dir"], max_rows=max_rows)
    x_train_df, y_train, _ = split_xy(train_frame, "binary_label")
    x_validation_df, y_validation, validation_meta = split_xy(validation_frame, "binary_label")
    x_test_df, y_test, test_meta = split_xy(test_frame, "binary_label")
    benign_train = x_train_df.loc[y_train.astype(int).to_numpy() == 0].reset_index(drop=True)
    if benign_train.empty:
        raise ValueError("Anomaly training requires benign training records")
    preprocessor = VCODAPreprocessor(scaler=config["preprocessing"].get("numeric_scaler", "robust"))
    x_benign = preprocessor.fit_transform(benign_train)
    x_validation = preprocessor.transform(x_validation_df)
    x_test = preprocessor.transform(x_test_df)
    names = models or [name for name in cfg.get("models", []) if name != "autoencoder" and name != "half_space_trees"]
    if not names:
        names = ["isolation_forest"]
    reports: dict[str, Any] = {}
    bundles: dict[str, AnomalyBundle] = {}
    skipped: dict[str, str] = {}
    numeric_baseline: dict[str, dict[str, float]] = {}
    for column in benign_train.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(benign_train[column], errors="coerce").dropna()
        if len(values):
            median = float(values.median())
            mad = float((values - median).abs().median())
            numeric_baseline[column] = {"median": median, "mad": max(mad, 1e-9)}
    for name in names:
        try:
            if name == "isolation_forest":
                model = IsolationForest(n_estimators=400, max_samples="auto", contamination="auto", random_state=42, n_jobs=-1)
                model.fit(x_benign)
            elif name == "local_outlier_factor":
                model = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1)
                model.fit(x_benign[: min(len(x_benign), 200_000)])
            elif name == "one_class_svm":
                model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
                model.fit(x_benign[: min(len(x_benign), 100_000)])
            else:
                raise ValueError(f"Unknown anomaly model: {name}")
            validation_scores = -np.asarray(model.score_samples(x_validation) if hasattr(model, "score_samples") else model.decision_function(x_validation))
            threshold = _threshold_from_validation(validation_scores, y_validation.astype(int).to_numpy(), float(cfg.get("target_benign_fpr", 0.01)))
            test_scores = -np.asarray(model.score_samples(x_test) if hasattr(model, "score_samples") else model.decision_function(x_test))
            low = float(np.quantile(validation_scores[y_validation.astype(int).to_numpy() == 0], 0.50))
            high = float(np.quantile(validation_scores[y_validation.astype(int).to_numpy() == 0], 0.999))
            confidence = np.clip((test_scores - low) / max(high - low, 1e-9), 0, 1)
            threshold_confidence = float(np.clip((threshold - low) / max(high - low, 1e-9), 0, 1))
            probabilities = np.column_stack([1 - confidence, confidence])
            metrics = evaluate_classification(y_test.astype(int).to_numpy(), probabilities, classes=["benign", "anomaly"], threshold=threshold_confidence)
            metrics.update({"raw_threshold": threshold, "confidence_threshold": threshold_confidence})
            bundle = AnomalyBundle(name, model, preprocessor, threshold, low, high, numeric_baseline)
            reports[name] = metrics
            bundles[name] = bundle
            validation_confidence = np.clip((validation_scores - low) / max(high - low, 1e-9), 0, 1)
            validation_predictions = validation_meta.copy()
            validation_predictions["actual"] = y_validation.astype(int).to_numpy()
            validation_predictions["anomaly_score"] = validation_scores
            validation_predictions["probability_attack"] = validation_confidence
            validation_predictions["prediction"] = (validation_scores >= threshold).astype(int)
            validation_predictions.to_parquet(resolve_path(f"artifacts/reports/anomaly_{name}_validation_predictions.parquet"), index=False)
            predictions = test_meta.copy()
            predictions["actual"] = y_test.astype(int).to_numpy()
            predictions["anomaly_score"] = test_scores
            predictions["anomaly_confidence"] = confidence
            predictions["prediction"] = (test_scores >= threshold).astype(int)
            predictions.to_parquet(resolve_path(f"artifacts/reports/anomaly_{name}_test_predictions.parquet"), index=False)
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
    if not reports:
        raise RuntimeError(f"No anomaly model trained successfully: {skipped}")
    best_name = max(reports, key=lambda key: float(reports[key].get("macro_f1", -1)))
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = resolve_path(f"models/anomaly/{best_name}_{version}.joblib")
    bundles[best_name].save(staging)
    manifest = load_manifest(config["dataset"]["prepared_dir"])
    registry = ModelRegistry()
    record = registry.register(
        model_name=best_name, version=version, task="anomaly_offline", artifact_path=staging,
        metrics=reports[best_name], dataset_fingerprint=manifest["dataset_fingerprint"],
        feature_list=preprocessor.schema.all_features if preprocessor.schema else [], preprocessing_version="1.0.0",
        hyperparameters={"benign_only": True}, threshold=bundles[best_name].threshold,
    )
    if promote_best:
        registry.promote(record["model_name"], record["version"])
    report = {"best_model": best_name, "models": reports, "skipped": skipped, "registry_record": record}
    report_path = "artifacts/reports/anomaly_training_report.json"
    dump_json(report, report_path)
    tracker = ExperimentTracker("anomaly_offline", backend="local")
    tracker.log_parameters({"best_model": best_name, "models": names, "benign_only": True})
    tracker.log_metrics({key: value for key, value in reports[best_name].items() if isinstance(value, (int, float))})
    tracker.log_artifact(report_path)
    tracker.log_artifact(staging)
    tracker.finish()
    return report


class OnlineAnomalyDetector:
    """River Half-Space Trees when installed; deterministic streaming z-score fallback otherwise."""

    def __init__(self, feature_names: list[str], seed: int = 42) -> None:
        self.feature_names = feature_names
        self.backend = "fallback"
        self.count = 0
        self.mean = {name: 0.0 for name in feature_names}
        self.m2 = {name: 0.0 for name in feature_names}
        self.model: Any = None
        if importlib.util.find_spec("river") is not None:
            from river import anomaly, compose, preprocessing
            self.model = compose.Pipeline(preprocessing.MinMaxScaler(), anomaly.HalfSpaceTrees(n_trees=25, height=10, window_size=500, seed=seed))
            self.backend = "river_half_space_trees"

    def score(self, record: dict[str, Any]) -> float:
        values = {name: float(record.get(name, 0.0) or 0.0) for name in self.feature_names}
        if self.model is not None:
            return float(self.model.score_one(values))
        if self.count < 2:
            return 0.0
        z_scores = []
        for name, value in values.items():
            variance = self.m2[name] / max(self.count - 1, 1)
            z_scores.append(abs(value - self.mean[name]) / max(variance ** 0.5, 1e-6))
        return float(np.tanh(max(z_scores, default=0.0) / 5.0))

    def learn(self, record: dict[str, Any], safe: bool = True) -> None:
        if not safe:
            return
        values = {name: float(record.get(name, 0.0) or 0.0) for name in self.feature_names}
        if self.model is not None:
            self.model.learn_one(values)
            return
        self.count += 1
        for name, value in values.items():
            delta = value - self.mean[name]
            self.mean[name] += delta / self.count
            self.m2[name] += delta * (value - self.mean[name])
