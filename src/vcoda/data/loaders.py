from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vcoda.utils.io import load_json, resolve_path

META_COLUMNS = {"row_id", "origin", "source_file", "sequence_group", "event_time", "binary_label", "attack_label"}


def load_manifest(processed_dir: str | Path = "data/processed") -> dict[str, Any]:
    manifest = load_json(Path(processed_dir) / "manifest.json")
    if not manifest:
        raise FileNotFoundError("Prepared manifest is missing. Run `vcoda prepare-data`.")
    return manifest


def load_split(split: str, processed_dir: str | Path = "data/processed", max_rows: int | None = None) -> pd.DataFrame:
    path = resolve_path(Path(processed_dir) / f"{split}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Prepared split is missing: {path}")
    frame = pd.read_parquet(path)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42)
    return frame.reset_index(drop=True)


def split_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} is missing")
    metadata = frame[[column for column in META_COLUMNS if column in frame.columns]].copy()
    features = frame.drop(columns=[column for column in META_COLUMNS if column in frame.columns], errors="ignore")
    return features, frame[target].copy(), metadata
