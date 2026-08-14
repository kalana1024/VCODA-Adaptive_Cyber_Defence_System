from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.util

import numpy as np

from vcoda.anomaly.models import AnomalyBundle
from vcoda.data.loaders import load_split, split_xy
from vcoda.models.bundle import ModelBundle
from vcoda.models.deep import DeepBundle
from vcoda.models.metrics import evaluate_classification
from vcoda.models.registry import ModelRegistry
from vcoda.utils.io import dump_json, load_yaml, resolve_path


def evaluate_active_models(config_path: str | Path = "configs/training.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    registry = ModelRegistry()
    test = load_split(
        "test",
        config["dataset"]["prepared_dir"],
        max_rows=config["supervised"].get("max_training_rows"),
    )
    reports: dict[str, Any] = {}
    binary_record = registry.active("supervised_binary")
    if binary_record:
        bundle = ModelBundle.load(resolve_path(binary_record["artifact_path"]), binary_record["artifact_sha256"])
        x, y, _ = split_xy(test, "binary_label")
        probabilities = bundle.predict_proba(x)
        y_values = y.astype(int).to_numpy()
        reports["supervised_binary"] = evaluate_classification(
            y_values, probabilities, classes=bundle.classes, threshold=bundle.threshold or 0.5
        )
        _save_evaluation_plots(
            "supervised_binary", y_values, probabilities, bundle.classes, bundle.threshold or 0.5
        )
    multiclass_record = registry.active("supervised_multiclass")
    if multiclass_record:
        bundle = ModelBundle.load(resolve_path(multiclass_record["artifact_path"]), multiclass_record["artifact_sha256"])
        x, y, _ = split_xy(test, "attack_label")
        known = y.astype(str).isin(set(bundle.classes))
        mapping = {value: index for index, value in enumerate(bundle.classes)}
        probabilities = bundle.predict_proba(x.loc[known])
        y_values = y.loc[known].astype(str).map(mapping).to_numpy()
        reports["supervised_multiclass"] = evaluate_classification(
            y_values, probabilities, classes=bundle.classes
        )
        _save_evaluation_plots("supervised_multiclass", y_values, probabilities, bundle.classes, 0.5)
    anomaly_record = registry.active("anomaly_offline")
    if anomaly_record:
        bundle = AnomalyBundle.load(resolve_path(anomaly_record["artifact_path"]), anomaly_record["artifact_sha256"])
        x, y, _ = split_xy(test, "binary_label")
        confidence = bundle.confidence(x)
        threshold_confidence = (bundle.threshold - bundle.score_low) / max(bundle.score_high - bundle.score_low, 1e-9)
        probabilities = np.column_stack([1 - confidence, confidence])
        y_values = y.astype(int).to_numpy()
        reports["anomaly_offline"] = evaluate_classification(
            y_values, probabilities, classes=["benign", "anomaly"], threshold=threshold_confidence
        )
        _save_evaluation_plots(
            "anomaly_offline", y_values, probabilities, ["benign", "anomaly"], threshold_confidence
        )
    deep_record = registry.active("deep_binary")
    if deep_record:
        bundle = DeepBundle.load(resolve_path(deep_record["artifact_path"]), deep_record["artifact_sha256"])
        if bundle.architecture == "mlp":
            x, y, _ = split_xy(test, "binary_label")
            probabilities = bundle.predict_proba(x, device="cpu")
            y_values = y.astype(int).to_numpy()
            reports["deep_binary"] = evaluate_classification(
                y_values, probabilities, classes=bundle.classes, threshold=bundle.threshold or 0.5
            )
            _save_evaluation_plots(
                "deep_binary", y_values, probabilities, bundle.classes, bundle.threshold or 0.5
            )
        else:
            reports["deep_binary"] = {
                "not_evaluated": "Temporal model evaluation requires sequence metadata and the same sequence builder used during training"
            }
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "models": reports}
    dump_json(report, "artifacts/reports/final_evaluation_report.json")
    _write_html(report, resolve_path("artifacts/reports/final_evaluation_report.html"))
    return report


def _write_html(report: dict[str, Any], path: Path) -> None:
    rows: list[str] = []
    for model, metrics in report["models"].items():
        values = [
            model,
            metrics.get("macro_precision", "-"),
            metrics.get("macro_recall", "-"),
            metrics.get("macro_f1", "-"),
            metrics.get("weighted_f1", "-"),
            metrics.get("false_positive_rate", "-"),
            metrics.get("roc_auc", "-"),
            metrics.get("pr_auc", "-"),
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>V-CODA Evaluation</title>"
        "<style>body{font-family:Arial;margin:2rem}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:.5rem}</style></head><body>"
        f"<h1>V-CODA Real Evaluation Report</h1><p>Generated {report['created_at']}. "
        "Values are produced only from the prepared held-out test partition.</p>"
        "<table><tr><th>Model</th><th>Macro precision</th><th>Macro recall</th>"
        "<th>Macro F1</th><th>Weighted F1</th><th>FPR</th><th>ROC-AUC</th><th>PR-AUC</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _save_evaluation_plots(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
) -> None:
    """Create real confusion-matrix and binary ROC/PR plots when matplotlib is installed."""
    if importlib.util.find_spec("matplotlib") is None:
        return
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

    plot_dir = resolve_path("artifacts/reports/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    predictions = (probabilities[:, 1] >= threshold).astype(int) if probabilities.shape[1] == 2 else probabilities.argmax(axis=1)

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, predictions, labels=list(range(len(classes))), display_labels=[str(value) for value in classes],
        xticks_rotation=45, cmap="Blues", ax=axis, colorbar=False,
    )
    figure.tight_layout()
    figure.savefig(plot_dir / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close(figure)

    if probabilities.shape[1] == 2 and len(np.unique(y_true)) > 1:
        figure, axis = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true, probabilities[:, 1], ax=axis, name=model_name)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{model_name}_roc_curve.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true, probabilities[:, 1], ax=axis, name=model_name)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{model_name}_precision_recall_curve.png", dpi=180)
        plt.close(figure)
