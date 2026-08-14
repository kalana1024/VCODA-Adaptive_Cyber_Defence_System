# Packaging Test Report

## Executed

- `python -m compileall -q src scripts dashboard tests` — passed.
- `python -m vcoda.cli --help` — passed.
- Editable package installation with the packaging environment's existing build tools — passed.
- Installed `vcoda --help` command — passed.
- `pytest -q` — **26 tests passed**.

Tests covered:

- derived flow features and schema compatibility;
- leakage detection;
- audit-chain tamper detection;
- upload/path validation;
- risk-score bounds;
- IOC extraction, local STIX/MISP loading and mocked external CTI adapters;
- probable MITRE mapping and graph correlation;
- protected-asset and allowlist response controls;
- ensemble combination, uncertainty and disagreement;
- MLP, CNN1D, CNN-LSTM, Transformer and autoencoder forward passes;
- drift-stream ingestion;
- model-bundle alignment and checksum validation;
- deliberate active-model corruption followed by self-healing rollback;
- API health, malformed input and PCAP upload rejection;
- inference performance smoke testing;
- expired response rollback de-duplication.

## Packaging environment

- Linux container
- Python 3.13
- CPU-only PyTorch available
- XGBoost, LightGBM, CatBoost, Optuna and SHAP importable

## Not executed

- NF-UQ-NIDS-v2 full scan, preprocessing, training or evaluation because the dataset was not supplied to the packaging runtime.
- CUDA training because no compatible GPU was available.
- Windows 10/11 setup scripts.
- Npcap live capture.
- TShark/Zeek extraction.
- Docker Desktop deployment.
- Windows Firewall changes or rollback.
- Live AbuseIPDB, VirusTotal or TAXII accounts; adapters were tested with mocked responses.
- Ruff static analysis because Ruff was not installed in the packaging runtime.

No trained model or model-performance claim is bundled. Real metrics are created only after the user runs training and evaluation on held-out real data.
