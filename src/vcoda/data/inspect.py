from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vcoda.data.common import count_csv_rows, discover_data_files, normalise_columns, sample_table
from vcoda.utils.io import dump_json, load_yaml, sha256_json


def _find_first(columns_original: list[str], candidates: list[Any]) -> str | None:
    normalised = {str(column).lower(): column for column in columns_original}
    for candidate in candidates:
        match = normalised.get(str(candidate).lower())
        if match is not None:
            return str(match)
    return None


def inspect_dataset(config_path: str | Path = "configs/training.yaml", *, full_scan: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    dataset = config["dataset"]
    files = discover_data_files(dataset["data_dir"], dataset["file_globs"])
    sample_rows = min(int(dataset.get("chunk_rows", 250_000)), 50_000)
    file_reports: list[dict[str, Any]] = []
    union_columns: Counter[str] = Counter()
    label_values: Counter[str] = Counter()
    attack_values: Counter[str] = Counter()
    origin_values: Counter[str] = Counter()
    detected: dict[str, str | None] = {"binary_label": None, "attack_label": None, "origin": None, "timestamp": None, "group": None}

    for path in files:
        frame_original = sample_table(path, rows=sample_rows)
        original_columns = [str(column) for column in frame_original.columns]
        frame = normalise_columns(frame_original)
        union_columns.update(frame.columns)
        binary_column = _find_first(original_columns, dataset.get("binary_label_candidates", []))
        attack_column = _find_first(original_columns, dataset.get("attack_label_candidates", []))
        origin_column = _find_first(original_columns, dataset.get("origin_candidates", []))
        timestamp_column = _find_first(original_columns, dataset.get("timestamp_candidates", []))
        group_column = _find_first(original_columns, dataset.get("group_candidates", []))
        for key, value in {
            "binary_label": binary_column, "attack_label": attack_column, "origin": origin_column,
            "timestamp": timestamp_column, "group": group_column,
        }.items():
            detected[key] = detected[key] or value
        if binary_column:
            label_values.update(frame_original[binary_column].astype(str).value_counts().to_dict())
        if attack_column:
            attack_values.update(frame_original[attack_column].astype(str).value_counts().to_dict())
        if origin_column:
            origin_values.update(frame_original[origin_column].astype(str).value_counts().to_dict())

        columns_report: list[dict[str, Any]] = []
        for original, normalised in zip(original_columns, frame.columns):
            series = frame[normalised]
            unique = int(series.nunique(dropna=True))
            rows = max(len(series), 1)
            columns_report.append({
                "original_name": original,
                "normalised_name": normalised,
                "dtype": str(series.dtype),
                "sample_null_fraction": float(series.isna().mean()),
                "sample_unique": unique,
                "sample_unique_ratio": float(unique / rows),
                "sample_constant": bool(unique <= 1),
                "sample_infinite": int(np.isinf(pd.to_numeric(series, errors="coerce")).sum()),
            })
        file_reports.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "rows": count_csv_rows(path) if full_scan else None,
            "sample_rows": len(frame),
            "columns": columns_report,
        })

    if detected["binary_label"] is None and detected["attack_label"] is None:
        raise ValueError("No configured binary or attack label column exists in the inspected files")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(Path(dataset["data_dir"]).resolve()),
        "file_count": len(files),
        "files": file_reports,
        "columns_present_in_files": dict(union_columns),
        "detected_columns": detected,
        "sample_binary_distribution": dict(label_values),
        "sample_attack_distribution": dict(attack_values),
        "sample_origin_distribution": dict(origin_values),
        "full_scan_row_counts": full_scan,
    }
    report["schema_fingerprint"] = sha256_json({
        "columns": sorted(union_columns), "detected": detected, "file_names": [path.name for path in files]
    })
    dump_json(report, dataset["schema_report"])
    return report
