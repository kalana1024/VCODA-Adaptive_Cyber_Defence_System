# NF-UQ-NIDS-v2 Dataset Engineering

## Exact placement

```text
data/raw/nf_uq_nids_v2/
```

Accepted input formats are CSV, CSV.GZ, and Parquet. The project does not include the dataset.

## Mandatory inspection

```powershell
vcoda inspect-data --data-dir ".\data\raw\nf_uq_nids_v2"
```

This reads actual file headers and samples, identifies available label/origin/time/group columns, records data types, null rates, sampled cardinality, constants, and a schema fingerprint. Preparation refuses to run when the schema report is absent.

## Preparation

```powershell
vcoda prepare-data
```

The process performs two passes:

1. Count real strata by origin, attack label, binary label, and split.
2. Deterministically sample, clean, derive safe flow features, and write train/validation/test Parquet files.

Outputs:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
data/processed/manifest.json
```

## Leakage controls

By default V-CODA excludes:

- binary and attack labels;
- dataset-origin fields;
- source and destination IP addresses;
- unique flow/session/record IDs;
- constant columns;
- sampled high-cardinality string identifiers;
- suspected post-event outcomes.

The exact exclusions are recorded in `manifest.json` and `leakage_report.json`.

## Split strategy

When the actual `Dataset`/origin field exists and values match configuration:

- training: NF-ToN-IoT-v2 and NF-BoT-IoT-v2;
- validation: NF-UNSW-NB15-v2;
- test: NF-CSE-CIC-IDS2018-v2.

Unmatched origins use a stable group-hash fallback. Adjust origin names only after checking the schema report.

## Memory profiles

Edit `configs/training.yaml`:

- 8–16 GB RAM: 250,000–500,000 rows per split.
- 16–32 GB RAM: 500,000–1,500,000 rows per split.
- 32 GB+: increase cautiously.

The full raw dataset is scanned in chunks, but training on all tens of millions of rows may exceed ordinary-PC memory. A source-separated stratified sample is usually more defensible than a leaking full random split.

## Temporal models

CNN/CNN-LSTM/Transformer training requires preserved sequence metadata. V-CODA stores `sequence_group` and `event_time` when actual source/group/time columns are present. If the data does not contain trustworthy order, temporal training stops with an error.
