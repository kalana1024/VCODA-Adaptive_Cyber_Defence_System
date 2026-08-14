from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from vcoda.data.common import discover_data_files, iter_table, normalise_columns
from vcoda.data.features import derive_flow_features
from vcoda.data.leakage import detect_leakage
from vcoda.utils.io import dump_json, load_json, load_yaml, resolve_path, sha256_file, sha256_json


def _actual_column(columns: list[str], candidates: list[Any]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if str(candidate).lower() in lookup:
            return lookup[str(candidate).lower()]
    return None


def _canonicalise(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    result = normalise_columns(frame)
    original_normalised = {str(column).lower(): normalised for column, normalised in zip(frame.columns, result.columns)}
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            key = str(candidate).lower()
            if key in original_normalised:
                source = original_normalised[key]
                if canonical != source and canonical not in result.columns:
                    result = result.rename(columns={source: canonical})
                resolved[canonical] = canonical
                break
            candidate_normalised = normalise_columns(pd.DataFrame(columns=[str(candidate)])).columns[0]
            if candidate_normalised in result.columns:
                if canonical != candidate_normalised and canonical not in result.columns:
                    result = result.rename(columns={candidate_normalised: canonical})
                resolved[canonical] = canonical
                break
    return result, resolved


def _binary_labels(series: pd.Series | None, attack_series: pd.Series | None, benign_values: list[Any]) -> pd.Series:
    benign = {str(value).strip().lower() for value in benign_values}
    if series is not None:
        raw = series.astype(str).str.strip().str.lower()
        numeric = pd.to_numeric(series, errors="coerce")
        labels = pd.Series(np.where(numeric.notna(), (numeric > 0).astype(int), (~raw.isin(benign)).astype(int)), index=series.index)
        return labels.astype(int)
    if attack_series is None:
        raise ValueError("Cannot create binary labels without a binary or attack label column")
    return (~attack_series.astype(str).str.strip().str.lower().isin(benign)).astype(int)


def _split_name(origin: pd.Series, group: pd.Series, split_config: dict[str, Any]) -> pd.Series:
    train_origins = {str(value).lower() for value in split_config.get("train_origins", [])}
    validation_origins = {str(value).lower() for value in split_config.get("validation_origins", [])}
    test_origins = {str(value).lower() for value in split_config.get("test_origins", [])}
    result = pd.Series(index=origin.index, dtype="object")
    origin_lower = origin.astype(str).str.lower()
    result[origin_lower.isin(train_origins)] = "train"
    result[origin_lower.isin(validation_origins)] = "validation"
    result[origin_lower.isin(test_origins)] = "test"
    unresolved = result.isna()
    if unresolved.any():
        train_fraction = float(split_config.get("fallback_train_fraction", 0.70))
        validation_fraction = float(split_config.get("fallback_validation_fraction", 0.15))
        hashed = group.astype(str).map(lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
        result[unresolved & (hashed < train_fraction)] = "train"
        result[unresolved & (hashed >= train_fraction) & (hashed < train_fraction + validation_fraction)] = "validation"
        result[unresolved & (hashed >= train_fraction + validation_fraction)] = "test"
    return result


def _stable_sample_mask(keys: pd.Series, probability: pd.Series) -> pd.Series:
    values = keys.astype(str).map(lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF))
    return values <= probability.clip(0.0, 1.0)


class _ParquetSplitWriter:
    def __init__(self, output_dir: Path) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Preparing large datasets requires pyarrow. Install requirements/training.txt") from exc
        self.pa = pa
        self.pq = pq
        self.output_dir = output_dir
        self.writers: dict[str, Any] = {}
        self.counts: Counter[str] = Counter()

    def write(self, split: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = self.pa.Table.from_pandas(frame, preserve_index=False)
        path = self.output_dir / f"{split}.parquet"
        if split not in self.writers:
            self.writers[split] = self.pq.ParquetWriter(path, table.schema, compression="zstd")
        self.writers[split].write_table(table)
        self.counts[split] += len(frame)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def prepare_dataset(
    training_config_path: str | Path = "configs/training.yaml",
    features_config_path: str | Path = "configs/features.yaml",
) -> dict[str, Any]:
    training = load_yaml(training_config_path)
    features_cfg = load_yaml(features_config_path)
    dataset_cfg = training["dataset"]
    schema_report = load_json(dataset_cfg["schema_report"])
    if features_cfg["schema_policy"].get("require_inspection_before_prepare", True) and not schema_report:
        raise RuntimeError("Run `vcoda inspect-data` before preparing the dataset")
    files = discover_data_files(dataset_cfg["data_dir"], dataset_cfg["file_globs"])
    output_dir = resolve_path(dataset_cfg["prepared_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    aliases = features_cfg["schema_policy"].get("canonical_aliases", {})
    chunk_rows = int(dataset_cfg.get("chunk_rows", 250_000))
    benign_values = dataset_cfg.get("benign_values", [0, "Benign"])
    split_cfg = dataset_cfg["split"]
    cap = int(dataset_cfg.get("sampling", {}).get("max_rows_per_class_per_origin", 150_000))

    # First pass obtains real stratum counts. No dataset columns are assumed without inspection.
    strata: Counter[tuple[str, str, int, str]] = Counter()
    column_resolution: dict[str, str] = {}
    first_clean_sample: pd.DataFrame | None = None
    file_hashes: dict[str, str] = {}
    for file_index, path in enumerate(files):
        file_hashes[str(path)] = sha256_file(path)
        for chunk_index, original in enumerate(iter_table(path, chunk_rows)):
            original_columns = [str(c) for c in original.columns]
            binary_original = _actual_column(original_columns, dataset_cfg.get("binary_label_candidates", []))
            attack_original = _actual_column(original_columns, dataset_cfg.get("attack_label_candidates", []))
            origin_original = _actual_column(original_columns, dataset_cfg.get("origin_candidates", []))
            group_original = _actual_column(original_columns, dataset_cfg.get("group_candidates", []))
            canonical, resolved = _canonicalise(original, aliases)
            column_resolution.update(resolved)
            binary_series = original[binary_original] if binary_original else None
            attack_series = original[attack_original] if attack_original else None
            labels = _binary_labels(binary_series, attack_series, benign_values)
            attacks = attack_series.astype(str) if attack_series is not None else pd.Series(np.where(labels == 1, "attack", "benign"), index=labels.index)
            origins = original[origin_original].astype(str) if origin_original else pd.Series(path.stem, index=labels.index)
            groups = original[group_original].astype(str) if group_original else pd.Series([f"{path.name}:{chunk_index}:{i}" for i in range(len(labels))], index=labels.index)
            splits = _split_name(origins, groups, split_cfg)
            for key, count in pd.DataFrame({"origin": origins, "attack": attacks, "label": labels, "split": splits}).value_counts().items():
                strata[(str(key[0]), str(key[1]), int(key[2]), str(key[3]))] += int(count)
            if first_clean_sample is None:
                first_clean_sample = canonical.head(50_000)

    if first_clean_sample is None:
        raise ValueError("Dataset files contained no rows")

    normalised_always_exclude = {str(v).strip().lower() for v in features_cfg["schema_policy"].get("always_exclude", [])}
    labels_lower = {str(v).strip().lower() for v in dataset_cfg.get("binary_label_candidates", []) + dataset_cfg.get("attack_label_candidates", []) + dataset_cfg.get("origin_candidates", [])}
    leakage = detect_leakage(
        first_clean_sample,
        label_columns=labels_lower,
        always_exclude=normalised_always_exclude,
        identifier_patterns=features_cfg["schema_policy"].get("identifier_patterns", []),
        high_cardinality_ratio=float(features_cfg["schema_policy"].get("high_cardinality_ratio", 0.98)),
    )
    excluded = set().union(*leakage.values())
    if features_cfg["schema_policy"].get("exclude_raw_identifiers", True):
        excluded.update({"src_ip", "dst_ip"})

    target_by_stratum = {key: min(count, cap) for key, count in strata.items()}
    writer = _ParquetSplitWriter(output_dir)
    retained_by_stratum: Counter[tuple[str, str, int, str]] = Counter()
    feature_columns: set[str] = set()
    try:
        for path in files:
            for chunk_index, original in enumerate(iter_table(path, chunk_rows)):
                original_columns = [str(c) for c in original.columns]
                binary_original = _actual_column(original_columns, dataset_cfg.get("binary_label_candidates", []))
                attack_original = _actual_column(original_columns, dataset_cfg.get("attack_label_candidates", []))
                origin_original = _actual_column(original_columns, dataset_cfg.get("origin_candidates", []))
                group_original = _actual_column(original_columns, dataset_cfg.get("group_candidates", []))
                timestamp_original = _actual_column(original_columns, dataset_cfg.get("timestamp_candidates", []))
                frame, _ = _canonicalise(original, aliases)
                binary_series = original[binary_original] if binary_original else None
                attack_series = original[attack_original] if attack_original else None
                labels = _binary_labels(binary_series, attack_series, benign_values)
                attacks = attack_series.astype(str) if attack_series is not None else pd.Series(np.where(labels == 1, "attack", "benign"), index=labels.index)
                origins = original[origin_original].astype(str) if origin_original else pd.Series(path.stem, index=labels.index)
                groups = original[group_original].astype(str) if group_original else pd.Series([f"{path.name}:{chunk_index}:{i}" for i in range(len(labels))], index=labels.index)
                splits = _split_name(origins, groups, split_cfg)
                meta = pd.DataFrame({
                    "binary_label": labels, "attack_label": attacks, "origin": origins,
                    "split": splits, "source_file": path.name,
                    "row_id": [hashlib.sha256(f"{path.name}:{chunk_index}:{i}".encode()).hexdigest()[:20] for i in range(len(frame))],
                    "sequence_group": groups.astype(str),
                    "event_time": (original[timestamp_original].astype(str) if timestamp_original else pd.Series(np.arange(len(frame)).astype(str), index=frame.index)),
                }, index=frame.index)
                frame = frame.drop(columns=[column for column in excluded if column in frame.columns], errors="ignore")
                for possible in [binary_original, attack_original, origin_original]:
                    if possible:
                        frame = frame.drop(columns=[normalise_columns(pd.DataFrame(columns=[possible])).columns[0]], errors="ignore")
                frame = derive_flow_features(frame, float(features_cfg["derived_features"].get("safe_division_epsilon", 1e-6)))
                frame = frame.replace([np.inf, -np.inf], np.nan).drop_duplicates()
                meta = meta.loc[frame.index]
                feature_columns.update(frame.columns)
                combined = pd.concat([meta.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
                stratum_keys = list(zip(combined["origin"].astype(str), combined["attack_label"].astype(str), combined["binary_label"].astype(int), combined["split"].astype(str)))
                probabilities = pd.Series([target_by_stratum[key] / max(strata[key], 1) for key in stratum_keys], index=combined.index)
                mask = _stable_sample_mask(combined["row_id"], probabilities)
                selected = combined.loc[mask]
                if selected.empty:
                    continue
                # Enforce exact caps after deterministic sampling.
                accepted_parts: list[pd.DataFrame] = []
                for key, part in selected.groupby(["origin", "attack_label", "binary_label", "split"], sort=False):
                    normal_key = (str(key[0]), str(key[1]), int(key[2]), str(key[3]))
                    remaining = target_by_stratum[normal_key] - retained_by_stratum[normal_key]
                    if remaining > 0:
                        accepted = part.head(remaining)
                        retained_by_stratum[normal_key] += len(accepted)
                        accepted_parts.append(accepted)
                if accepted_parts:
                    accepted_frame = pd.concat(accepted_parts, ignore_index=True)
                    for split, split_frame in accepted_frame.groupby("split", sort=False):
                        writer.write(str(split), split_frame.drop(columns=["split"]))
    finally:
        writer.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_fingerprint": schema_report.get("schema_fingerprint"),
        "dataset_fingerprint": sha256_json(file_hashes),
        "file_hashes": file_hashes,
        "column_resolution": column_resolution,
        "leakage_report": leakage,
        "excluded_columns": sorted(excluded),
        "feature_columns": sorted(feature_columns - {"binary_label", "attack_label", "origin", "source_file", "row_id"}),
        "metadata_columns": ["row_id", "origin", "source_file", "sequence_group", "event_time"],
        "target_columns": ["binary_label", "attack_label"],
        "strata_source_counts": {"|".join(map(str, key)): value for key, value in strata.items()},
        "strata_retained_counts": {"|".join(map(str, key)): value for key, value in retained_by_stratum.items()},
        "split_rows": dict(writer.counts),
        "outputs": {split: str(output_dir / f"{split}.parquet") for split in writer.counts},
    }
    dump_json(manifest, output_dir / "manifest.json")
    dump_json(leakage, "artifacts/reports/leakage_report.json")
    return manifest
