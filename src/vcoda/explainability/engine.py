from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vcoda.models.bundle import ModelBundle
from vcoda.utils.io import dump_json


class ExplainabilityEngine:
    def explain_supervised(self, bundle: ModelBundle, frame: pd.DataFrame, top_k: int = 8) -> list[dict[str, Any]]:
        transformed = bundle.preprocessor.transform(frame)
        feature_names = bundle.preprocessor.feature_names()
        model = bundle.model
        if importlib.util.find_spec("shap") is not None and hasattr(model, "get_booster"):
            import shap
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(transformed[:1])
            array = np.asarray(values)
            if array.ndim == 3:
                predicted = int(bundle.predict(frame[:1])[0])
                contributions = array[0, :, predicted]
            elif array.ndim == 2:
                contributions = array[0]
            else:
                contributions = np.ravel(array)
            ranked = np.argsort(np.abs(contributions))[::-1][:top_k]
            return [{"feature": feature_names[index], "contribution": float(contributions[index]), "method": "shap"} for index in ranked]
        if hasattr(model, "coef_"):
            predicted = int(bundle.predict(frame[:1])[0])
            coefficients = np.asarray(model.coef_)
            vector = coefficients[0 if coefficients.shape[0] == 1 else predicted]
            contributions = vector * transformed[0]
            ranked = np.argsort(np.abs(contributions))[::-1][:top_k]
            return [{"feature": feature_names[index], "contribution": float(contributions[index]), "method": "linear_contribution"} for index in ranked]
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
            ranked = np.argsort(importance)[::-1][:top_k]
            return [{"feature": feature_names[index], "global_importance": float(importance[index]), "method": "global_importance_only"} for index in ranked]
        return [{"method": "not_available", "reason": "The active model does not expose a supported faithful explanation interface"}]

    def save(self, event_id: str, explanation: dict[str, Any]) -> Path:
        path = Path("artifacts/explanations") / f"{event_id}.json"
        dump_json(explanation, path)
        return path
