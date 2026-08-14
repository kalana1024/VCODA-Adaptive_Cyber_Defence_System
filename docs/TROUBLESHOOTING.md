# Windows Troubleshooting

## `vcoda` is not recognised

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or run:

```powershell
.\.venv\Scripts\vcoda.exe system-check
```

## Dataset files not found

Check:

```powershell
Get-ChildItem .\data\raw\nf_uq_nids_v2 -Recurse
```

Extract archives first. Supported operational input is CSV, CSV.GZ, or Parquet.

## `pyarrow` error

```powershell
python -m pip install pyarrow
```

## Out of memory

Reduce in `configs/training.yaml`:

```yaml
sampling:
  max_rows_per_split: 300000
  max_rows_per_class_per_origin: 50000
supervised:
  max_training_rows: 300000
```

Reduce deep batch size to 256 or 128.

## XGBoost/LightGBM/CatBoost install error

Upgrade pip, then install Visual C++ Build Tools if a wheel is unavailable:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install xgboost lightgbm catboost
```

## CUDA is false

Verify `nvidia-smi`, remove CPU-only PyTorch, and install the correct official CUDA wheel. V-CODA still works on CPU.

## Temporal model says no valid sequences

Do not disable the safety check blindly. Confirm the actual dataset has a reliable timestamp and grouping field, then set `timestamp_column` and `sequence_group_column` after inspection.

## Live capture sees no packets

- Install Npcap.
- Verify the interface name with Scapy.
- Run PowerShell as Administrator.
- Check VPN/virtual adapter selection.
- Ensure another capture driver is not blocking access.

## API returns no active model

Train and promote the binary model:

```powershell
vcoda train supervised --task binary
vcoda list-models
```

## Self-healing reports manual recovery required

No checksum-valid historical model is available. Retrain or manually promote a known-good registered version.
