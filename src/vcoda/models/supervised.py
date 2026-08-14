from __future__ import annotations

import importlib.util
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from vcoda.data.loaders import load_manifest, load_split, split_xy
from vcoda.models.bundle import ModelBundle
from vcoda.models.calibration import ProbabilityCalibrator
from vcoda.models.metrics import benchmark_inference, choose_binary_threshold, evaluate_classification
from vcoda.models.preprocessing import VCODAPreprocessor
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.experiments import ExperimentTracker
from vcoda.utils.io import dump_json, load_yaml, resolve_path


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _model_factory(name: str, task: str, classes: int, seed: int, n_jobs: int, params: dict[str, Any] | None = None) -> Any:
    params = params or {}
    if name == "xgboost":
        if not _available("xgboost"):
            raise RuntimeError("XGBoost is not installed. Install requirements/training.txt")
        from xgboost import XGBClassifier
        base = {
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.06,
            "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2,
            "reg_alpha": 0.05, "reg_lambda": 1.0, "tree_method": "hist",
            "random_state": seed, "n_jobs": n_jobs, "eval_metric": "logloss",
        }
        if task == "multiclass":
            base.update({"objective": "multi:softprob", "num_class": classes, "eval_metric": "mlogloss"})
        else:
            base.update({"objective": "binary:logistic"})
        base.update(params)
        return XGBClassifier(**base)
    if name == "lightgbm":
        if not _available("lightgbm"):
            raise RuntimeError("LightGBM is not installed")
        from lightgbm import LGBMClassifier
        base = {
            "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
            "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.05,
            "reg_lambda": 1.0, "random_state": seed, "n_jobs": n_jobs, "verbosity": -1,
            "objective": "multiclass" if task == "multiclass" else "binary",
        }
        if task == "multiclass":
            base["num_class"] = classes
        base.update(params)
        return LGBMClassifier(**base)
    if name == "catboost":
        if not _available("catboost"):
            raise RuntimeError("CatBoost is not installed")
        from catboost import CatBoostClassifier
        base = {
            "iterations": 500, "depth": 8, "learning_rate": 0.06,
            "loss_function": "MultiClass" if task == "multiclass" else "Logloss",
            "random_seed": seed, "verbose": False, "thread_count": n_jobs,
            "allow_writing_files": False,
        }
        base.update(params)
        return CatBoostClassifier(**base)
    if name == "random_forest":
        base = {"n_estimators": 350, "max_depth": None, "min_samples_leaf": 2,
                "class_weight": "balanced_subsample", "random_state": seed, "n_jobs": n_jobs}
        base.update(params)
        return RandomForestClassifier(**base)
    if name == "extra_trees":
        base = {"n_estimators": 350, "max_depth": None, "min_samples_leaf": 2,
                "class_weight": "balanced", "random_state": seed, "n_jobs": n_jobs}
        base.update(params)
        return ExtraTreesClassifier(**base)
    raise ValueError(f"Unknown supervised model: {name}")


def _tune_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    task: str,
    classes: int,
    trials: int,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    if trials <= 0 or not _available("optuna"):
        return {}
    import optuna
    from sklearn.metrics import f1_score

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }
        model = _model_factory("xgboost", task, classes, seed, n_jobs, params)
        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_validation, y_validation)], verbose=False)
        predicted = model.predict(x_validation)
        return float(f1_score(y_validation, predicted, average="macro", zero_division=0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


def _save_global_explanations(
    bundle: ModelBundle,
    validation_frame: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    max_rows: int = 2000,
) -> dict[str, Any]:
    """Save faithful global importance and SHAP summary artefacts when supported."""
    output_dir = resolve_path("artifacts/reports/explanations")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = validation_frame.head(max_rows)
    transformed = bundle.preprocessor.transform(sample)
    names = bundle.preprocessor.feature_names()
    result: dict[str, Any] = {"sample_rows": len(sample), "method": None, "outputs": []}
    model = bundle.model

    importance: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        result["method"] = "model_feature_importance"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importance = np.mean(np.abs(coefficients), axis=0)
        result["method"] = "absolute_linear_coefficient"
    if importance is not None and len(importance) == len(names):
        table = pd.DataFrame({"feature": names, "importance": importance}).sort_values(
            "importance", ascending=False
        )
        table_path = output_dir / f"{task}_{model_name}_global_importance.csv"
        table.to_csv(table_path, index=False)
        result["outputs"].append(str(table_path))

    if importlib.util.find_spec("shap") is not None and hasattr(model, "get_booster"):
        try:
            import matplotlib.pyplot as plt
            import shap

            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(transformed)
            plot_path = output_dir / f"{task}_{model_name}_shap_summary.png"
            shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=25)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=180, bbox_inches="tight")
            plt.close()
            result["method"] = "shap_tree"
            result["outputs"].append(str(plot_path))
        except Exception as exc:
            result["shap_error"] = f"{type(exc).__name__}: {exc}"
    return result


def train_supervised(
    config_path: str | Path = "configs/training.yaml",
    *,
    task: str = "binary",
    model_names: list[str] | None = None,
    optuna_trials: int | None = None,
    promote_best: bool = True,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    supervised = config["supervised"]
    seed = 42
    max_rows = supervised.get("max_training_rows")
    n_jobs = int(supervised.get("n_jobs", -1))
    manifest = load_manifest(config["dataset"]["prepared_dir"])
    train_frame = load_split("train", config["dataset"]["prepared_dir"], max_rows=max_rows)
    validation_frame = load_split("validation", config["dataset"]["prepared_dir"], max_rows=max_rows)
    test_frame = load_split("test", config["dataset"]["prepared_dir"], max_rows=max_rows)
    target = "binary_label" if task == "binary" else "attack_label"
    x_train_df, y_train_raw, train_meta = split_xy(train_frame, target)
    x_validation_df, y_validation_raw, validation_meta = split_xy(validation_frame, target)
    x_test_df, y_test_raw, test_meta = split_xy(test_frame, target)

    label_encoder: LabelEncoder | None = None
    if task == "multiclass":
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train_raw.astype(str))
        classes = label_encoder.classes_.tolist()
        known = set(classes)

        val_mask = y_validation_raw.astype(str).isin(known)
        if not val_mask.any():
            raise RuntimeError("No validation rows have attack labels present in training set")
        x_validation_df = x_validation_df.loc[val_mask].reset_index(drop=True)
        validation_meta = validation_meta.loc[val_mask].reset_index(drop=True)
        y_validation_raw = y_validation_raw.loc[val_mask].reset_index(drop=True)

        test_mask = y_test_raw.astype(str).isin(known)
        if not test_mask.any():
            raise RuntimeError("No test rows have attack labels present in training set")
        x_test_df = x_test_df.loc[test_mask].reset_index(drop=True)
        test_meta = test_meta.loc[test_mask].reset_index(drop=True)
        y_test_raw = y_test_raw.loc[test_mask].reset_index(drop=True)

        y_train = label_encoder.transform(y_train_raw.astype(str))
        y_validation = label_encoder.transform(y_validation_raw.astype(str))
        y_test = label_encoder.transform(y_test_raw.astype(str))
    else:
        y_train = y_train_raw.astype(int).to_numpy()
        y_validation = y_validation_raw.astype(int).to_numpy()
        y_test = y_test_raw.astype(int).to_numpy()
        classes = ["benign", "attack"]

    preprocessor = VCODAPreprocessor(scaler=config["preprocessing"].get("numeric_scaler", "robust"))
    x_train = preprocessor.fit_transform(x_train_df)
    x_validation = preprocessor.transform(x_validation_df)
    x_test = preprocessor.transform(x_test_df)

    names = model_names or [supervised["primary_model"], *supervised.get("comparison_models", [])]
    names = list(dict.fromkeys(names))
    reports: dict[str, Any] = {}
    trained_bundles: dict[str, ModelBundle] = {}
    skipped: dict[str, str] = {}
    for name in names:
        try:
            parameters: dict[str, Any] = {}
            if name == "xgboost":
                parameters = _tune_xgboost(
                    x_train, y_train, x_validation, y_validation, task=task,
                    classes=len(classes), trials=int(optuna_trials if optuna_trials is not None else supervised.get("optuna_trials", 0)),
                    seed=seed, n_jobs=n_jobs,
                )
            model = _model_factory(name, task, len(classes), seed, n_jobs, parameters)
            weights = compute_sample_weight("balanced", y_train)
            fit_kwargs: dict[str, Any] = {"sample_weight": weights}
            if name in {"xgboost", "lightgbm"}:
                fit_kwargs["eval_set"] = [(x_validation, y_validation)]
                if name == "xgboost":
                    fit_kwargs["verbose"] = False
                elif name == "lightgbm":
                    fit_kwargs["callbacks"] = []
                    try:
                        from lightgbm import early_stopping, log_evaluation
                        fit_kwargs["callbacks"] = [log_evaluation(period=-1)]
                    except ImportError:
                        pass
            try:
                model.fit(x_train, y_train, **fit_kwargs)
            except TypeError:
                fit_kwargs.pop("sample_weight", None)
                model.fit(x_train, y_train, **fit_kwargs)
            validation_raw = model.predict_proba(x_validation)
            calibrator = ProbabilityCalibrator().fit(validation_raw, y_validation) if supervised.get("calibrate", True) else None
            validation_probabilities = calibrator.transform(validation_raw) if calibrator else validation_raw
            threshold = choose_binary_threshold(y_validation, validation_probabilities[:, 1]) if task == "binary" else None
            test_raw = model.predict_proba(x_test)
            test_probabilities = calibrator.transform(test_raw) if calibrator else test_raw
            metrics = evaluate_classification(y_test, test_probabilities, classes=classes, threshold=threshold or 0.5)
            metrics["inference"] = benchmark_inference(model.predict_proba, x_test[: min(len(x_test), 10000)])
            metrics["model"] = name
            metrics["task"] = task
            metrics["hyperparameters"] = parameters
            bundle = ModelBundle(
                name=name, task=task, model=model, preprocessor=preprocessor, classes=[str(c) for c in classes],
                threshold=threshold, calibrator=calibrator,
                metadata={"dataset_fingerprint": manifest["dataset_fingerprint"], "trained_at": datetime.now(timezone.utc).isoformat()},
            )
            trained_bundles[name] = bundle
            reports[name] = metrics
            validation_prediction_frame = validation_meta.copy()
            validation_prediction_frame["actual"] = y_validation
            validation_prediction_frame["prediction"] = (validation_probabilities[:, 1] >= (threshold or 0.5)).astype(int) if task == "binary" else validation_probabilities.argmax(axis=1)
            for index, class_name in enumerate(classes):
                validation_prediction_frame[f"probability_{class_name}"] = validation_probabilities[:, index]
            validation_prediction_path = resolve_path(f"artifacts/reports/{task}_{name}_validation_predictions.parquet")
            validation_prediction_path.parent.mkdir(parents=True, exist_ok=True)
            validation_prediction_frame.to_parquet(validation_prediction_path, index=False)
            prediction_frame = test_meta.copy()
            prediction_frame["actual"] = y_test
            prediction_frame["prediction"] = bundle.predict(x_test_df)
            for index, class_name in enumerate(classes):
                prediction_frame[f"probability_{class_name}"] = test_probabilities[:, index]
            prediction_path = resolve_path(f"artifacts/reports/{task}_{name}_test_predictions.parquet")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_frame.to_parquet(prediction_path, index=False)
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"

    if not reports:
        raise RuntimeError(f"No supervised model trained successfully: {skipped}")
    selection_metric = supervised.get("selection_metric", "macro_f1")
    best_name = max(reports, key=lambda key: float(reports[key].get(selection_metric, -math.inf)))
    best_bundle = trained_bundles[best_name]
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging_path = resolve_path(f"models/supervised/{task}_{best_name}_{version}.joblib")
    best_bundle.save(staging_path)
    registry = ModelRegistry()
    record = registry.register(
        model_name=f"{task}_{best_name}", version=version, task=f"supervised_{task}", artifact_path=staging_path,
        metrics=reports[best_name], dataset_fingerprint=manifest["dataset_fingerprint"],
        feature_list=preprocessor.schema.all_features if preprocessor.schema else [], preprocessing_version="1.0.0",
        hyperparameters=reports[best_name].get("hyperparameters", {}), threshold=best_bundle.threshold,
    )
    if promote_best:
        registry.promote(record["model_name"], record["version"])
    global_explanations = _save_global_explanations(
        best_bundle, x_validation_df, task=task, model_name=best_name
    )
    report = {
        "task": task, "best_model": best_name, "best_registry_record": record,
        "models": reports, "skipped_models": skipped, "classes": classes,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "global_explanations": global_explanations,
    }
    report_path = f"artifacts/reports/supervised_{task}_training_report.json"
    dump_json(report, report_path)
    tracker = ExperimentTracker(f"supervised_{task}", backend="local")
    tracker.log_parameters({"best_model": best_name, "models": names, "dataset_fingerprint": manifest["dataset_fingerprint"]})
    tracker.log_metrics({key: value for key, value in reports[best_name].items() if isinstance(value, (int, float))})
    tracker.log_artifact(report_path)
    tracker.log_artifact(staging_path)
    tracker.finish()
    if label_encoder is not None:
        encoder_path = resolve_path(f"artifacts/encoders/{task}_label_encoder.joblib")
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(label_encoder, encoder_path)
    return report
