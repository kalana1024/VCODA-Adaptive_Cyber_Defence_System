from __future__ import annotations

import re
from typing import Any

import pandas as pd


def detect_leakage(
    frame: pd.DataFrame,
    *,
    label_columns: set[str],
    always_exclude: set[str],
    identifier_patterns: list[str],
    high_cardinality_ratio: float,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "labels": [], "configured": [], "identifiers": [], "high_cardinality": [],
        "constants": [], "post_event_suspects": [],
    }
    rows = max(len(frame), 1)
    for column in frame.columns:
        lower = column.lower()
        unique = int(frame[column].nunique(dropna=True))
        if lower in label_columns:
            result["labels"].append(column)
        if lower in always_exclude:
            result["configured"].append(column)
        if any(re.search(pattern, lower) for pattern in identifier_patterns):
            result["identifiers"].append(column)
        if unique <= 1:
            result["constants"].append(column)
        if unique / rows >= high_cardinality_ratio and (
            pd.api.types.is_string_dtype(frame[column]) or pd.api.types.is_object_dtype(frame[column])
        ):
            result["high_cardinality"].append(column)
        if any(token in lower for token in ["ground_truth", "is_attack", "attack_id", "response_action", "incident_outcome"]):
            result["post_event_suspects"].append(column)
    return {key: sorted(set(value)) for key, value in result.items()}
