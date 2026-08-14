from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler, StandardScaler


SENTINEL_ABS_THRESHOLD = 1e15
"""Some raw NetFlow exports (e.g. *_SECOND_BYTES when duration is zero) encode
'not applicable' as a huge placeholder like 1e30 instead of a null. Left alone,
that single value dominates that column's variance/IQR and poisons anything
built on covariance (RobustScaler, CORAL, BatchNorm batch statistics). No real
flow byte/packet/duration count approaches even 1e15, so anything past that is
treated as missing and handed to the existing median imputer instead."""


@dataclass
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    all_features: list[str]


def _sanitize_sentinels(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in numeric_columns:
        if column not in sanitized.columns:
            continue
        values = pd.to_numeric(sanitized[column], errors="coerce")
        sanitized[column] = values.where(values.abs() <= SENTINEL_ABS_THRESHOLD, np.nan)
    return sanitized


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """Standard heavy-tail compression for zero-inflated NetFlow counters (byte/packet
    counts, durations, throughput): monotonic and sign-preserving, so ordering and
    outlier direction survive, but a rare million-byte retransmission burst no longer
    dwarfs every other feature's scale after RobustScaler."""
    return np.sign(values) * np.log1p(np.abs(values))


class VCODAPreprocessor:
    def __init__(self, scaler: str = "robust") -> None:
        self.scaler_name = scaler
        self.pipeline: ColumnTransformer | None = None
        self.schema: FeatureSchema | None = None

    def fit(self, frame: pd.DataFrame) -> "VCODAPreprocessor":
        clean = frame.copy()
        numeric = clean.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical = [column for column in clean.columns if column not in numeric]
        clean = _sanitize_sentinels(clean, numeric)
        scaler = RobustScaler(quantile_range=(5, 95)) if self.scaler_name == "robust" else StandardScaler()
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("log1p", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one")),
            ("scaler", scaler),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
        ])
        self.pipeline = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ], remainder="drop", verbose_feature_names_out=True)
        self.pipeline.fit(clean)
        self.schema = FeatureSchema(numeric=numeric, categorical=categorical, all_features=clean.columns.tolist())
        return self

    def align(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self.schema is None:
            raise RuntimeError("Preprocessor has not been fitted")
        aligned = frame.copy()
        missing = [column for column in self.schema.all_features if column not in aligned.columns]
        extra = [column for column in aligned.columns if column not in self.schema.all_features]
        for column in missing:
            aligned[column] = np.nan
        aligned = aligned[self.schema.all_features]
        coverage = (len(self.schema.all_features) - len(missing)) / max(len(self.schema.all_features), 1)
        return aligned, {"missing": missing, "extra": extra, "coverage": coverage}

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None or self.schema is None:
            raise RuntimeError("Preprocessor has not been fitted")
        aligned, _ = self.align(frame)
        aligned = _sanitize_sentinels(aligned, self.schema.numeric)
        transformed = self.pipeline.transform(aligned)
        transformed = np.nan_to_num(
            transformed,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return np.clip(transformed, -1e6, 1e6).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    def feature_names(self) -> list[str]:
        if self.pipeline is None:
            return []
        return [str(value) for value in self.pipeline.get_feature_names_out()]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "VCODAPreprocessor":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Loaded object is not a VCODAPreprocessor")
        return loaded
