from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: list[str] | list[int],
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    probabilities = np.nan_to_num(np.asarray(probabilities), nan=0.0, posinf=1.0, neginf=0.0)
    if probabilities.ndim == 1 or probabilities.shape[1] == 2:
        positive = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
        positive = np.clip(positive, 0.0, 1.0)
        predictions = (positive >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        result: dict[str, Any] = {
            "accuracy": accuracy_score(y_true, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
            "macro_precision": precision_score(y_true, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_true, predictions),
            "roc_auc": roc_auc_score(y_true, positive) if len(np.unique(y_true)) > 1 else None,
            "pr_auc": _pr_auc(y_true, positive) if len(np.unique(y_true)) > 1 else None,
            "brier_score": brier_score_loss(y_true, positive),
            "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(fn + tp, 1),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": threshold,
        }
    else:
        predictions = probabilities.argmax(axis=1)
        roc_auc_val = None
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc_val = roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=list(range(probabilities.shape[1])))
            except ValueError:
                roc_auc_val = None
        result = {
            "accuracy": accuracy_score(y_true, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
            "macro_precision": precision_score(y_true, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_true, predictions),
            "roc_auc": roc_auc_val,
            "pr_auc": None,
            "confusion_matrix": confusion_matrix(y_true, predictions).astype(int).tolist(),
            "threshold": None,
        }
    result["classification_report"] = classification_report(
        y_true, predictions, labels=list(range(len(classes))), target_names=[str(v) for v in classes],
        zero_division=0, output_dict=True,
    )
    return _to_python(result)


def choose_binary_threshold(y_true: np.ndarray, positive_probabilities: np.ndarray, max_fpr: float = 0.05) -> float:
    positive_probabilities = np.nan_to_num(np.asarray(positive_probabilities), nan=0.0, posinf=1.0, neginf=0.0)
    positive_probabilities = np.clip(positive_probabilities, 0.0, 1.0)
    candidates = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        predicted = (positive_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        score = f1_score(y_true, predicted, average="macro", zero_division=0)
        if fpr <= max_fpr and score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def benchmark_inference(predict: Callable[[np.ndarray], np.ndarray], data: np.ndarray, repeats: int = 3) -> dict[str, float]:
    if len(data) == 0:
        return {"latency_ms_per_row": 0.0, "throughput_rows_per_second": 0.0}
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        predict(data)
        elapsed.append(time.perf_counter() - started)
    seconds = min(elapsed)
    return {
        "latency_ms_per_row": seconds * 1000 / len(data),
        "throughput_rows_per_second": len(data) / max(seconds, 1e-9),
    }


def _pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    order = np.argsort(recall)
    return float(auc(recall[order], precision[order]))


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
