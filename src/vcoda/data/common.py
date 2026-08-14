from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path
from typing import Iterator

import pandas as pd


def normalise_name(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [normalise_name(str(column)) for column in result.columns]
    return result


def discover_data_files(data_dir: str | Path, globs: list[str]) -> list[Path]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")
    files: list[Path] = []
    for pattern in globs:
        files.extend(path for path in base.rglob(pattern) if path.is_file())
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError(f"No supported dataset files found in {base}")
    return unique


def iter_table(path: str | Path, chunk_rows: int = 250_000) -> Iterator[pd.DataFrame]:
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pyarrow") from exc
        parquet = pq.ParquetFile(target)
        for batch in parquet.iter_batches(batch_size=chunk_rows):
            yield batch.to_pandas()
        return
    if lower.endswith(".csv") or lower.endswith(".csv.gz") or lower.endswith(".gz"):
        yield from pd.read_csv(target, chunksize=chunk_rows, low_memory=False)
        return
    raise ValueError(f"Unsupported data file: {target}")


def sample_table(path: str | Path, rows: int = 10_000) -> pd.DataFrame:
    target = Path(path)
    if target.name.lower().endswith(".parquet"):
        try:
            return pd.read_parquet(target).head(rows)
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pyarrow") from exc
    return pd.read_csv(target, nrows=rows, low_memory=False)


def count_csv_rows(path: str | Path) -> int | None:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(target).metadata.num_rows)
        except Exception:
            return None
    opener = gzip.open if target.name.lower().endswith(".gz") else open
    try:
        with opener(target, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except Exception:
        return None
