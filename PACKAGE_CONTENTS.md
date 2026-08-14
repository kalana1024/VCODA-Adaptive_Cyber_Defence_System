# Package Contents

This source-only package contains the complete V-CODA implementation, configuration, tests,
Windows helper scripts and documentation. It intentionally excludes the NF-UQ-NIDS-v2 dataset,
API keys and trained model files.

## Main operational components

- chunked CSV/CSV.GZ/Parquet inspection and preparation;
- binary and multiclass supervised training with XGBoost and comparison models;
- PyTorch MLP, CNN1D, CNN-LSTM, Transformer and autoencoder training;
- offline and online anomaly detection;
- validation-trained ensemble fusion;
- MITRE ATT&CK probable mapping and NetworkX incident graph;
- local IOC, STIX, MISP and optional AbuseIPDB/VirusTotal/TAXII adapters;
- drift monitoring and controlled self-healing;
- SHAP/local explanations and real evaluation plots;
- FastAPI backend, Streamlit dashboard and Typer CLI;
- PCAP analysis, live Npcap/Scapy monitoring and feature-compatibility reporting;
- response recommendation, optional human-approved Windows Firewall action and rollback;
- local model registry, experiment tracking and tamper-evident audit logging.

See `PROJECT_MANIFEST.json` and `MANIFEST.sha256` for the packaged file inventory and checksums.
