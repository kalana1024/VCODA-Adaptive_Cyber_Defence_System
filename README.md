# V-CODA

**Verifiable Cybersecurity-Oriented Detection and Adaptive Response Architecture**

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![Status](https://img.shields.io/badge/status-research-orange)

V-CODA is a research-grade, production-style implementation of a hybrid network intrusion
detection and adaptive response platform. It combines gradient-boosted and deep classifiers,
unsupervised anomaly detection, validation-trained ensemble fusion, MITRE ATT&CK reasoning, and
an audit-chained self-healing layer — with **no falsely pretrained model** and **no bundled
dataset**. You train it on your own data and it tells you honestly what it can and cannot do.

> Built for [NF-UQ-NIDS-v2](https://research.unsw.edu.au/projects/nf-uq-nids-v2). No dataset,
> pretrained weights, or performance claims ship with this repository.

---

## Table of contents

- [Why V-CODA](#why-v-coda)
- [Detection architecture](#detection-architecture)
- [Quick start (Windows)](#quick-start-windows)
- [Project structure](#project-structure)
- [Safety and self-healing](#safety-and-self-healing)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Why V-CODA

Most academic NIDS prototypes stop at "trained a classifier, got an F1 score." V-CODA is built
around the parts that make a detector usable in a real operational loop:

- **Multiple learners, one decision** — tree ensembles, deep sequence models, and unsupervised
  anomaly detectors are fused with a validation-trained combiner, not hand-picked thresholds.
- **Explainable and traceable** — SHAP attributions where supported, probable MITRE ATT&CK
  mapping, and an append-only SHA-256 audit chain for every decision and response action.
- **Adaptive, not reckless** — drift is monitored (ADWIN / Page-Hinkley) and anomaly models can
  update online; the supervised model never silently retrains or self-promotes.
- **Safe by default** — response actions are recommendation-only unless explicitly configured,
  confirmed, and run with administrator privileges.
- **Self-healing** — corrupted model artefacts are quarantined and rolled back to the last
  checksum-valid version automatically.

## Detection architecture

```text
CSV / Parquet / PCAP / live packets / CTI
                    |
          schema and security validation
                    |
       feature compatibility and preprocessing
                    |
    +---------------+----------------+----------------+
    |               |                |                |
 XGBoost        Deep models     Anomaly models    CTI/rules
 binary/multi   MLP/CNN/LSTM    Isolation/AE/HST  IOC/STIX
    |               |                |                |
    +---------------+----------------+----------------+
                    |
       validation-trained ensemble fusion
                    |
   uncertainty + disagreement + risk calculation
                    |
       probable MITRE ATT&CK reasoning
                    |
          incident/threat graph correlation
                    |
     deterministic response-policy verification
                    |
 recommendation / simulation / approved action
                    |
        audit chain + registry + drift monitor
                    |
       checksum/rollback self-healing manager
```

| Layer | Components |
|---|---|
| **Known-attack models** | XGBoost (primary), LightGBM, CatBoost, Random Forest, Extra Trees |
| **Deep models** | PyTorch MLP, 1D-CNN, CNN-LSTM, Transformer encoder, benign-only autoencoder |
| **Anomaly models** | Isolation Forest, optional LOF/One-Class SVM, autoencoder, River Half-Space Trees |
| **Fusion** | Validation-trained logistic stacking or validation-optimised weighted soft voting |
| **Threat reasoning** | Probable MITRE ATT&CK mapping, IOC/STIX enrichment, NetworkX incident graph |
| **Adaptation** | ADWIN/Page-Hinkley drift monitoring, safe online anomaly updates |
| **Response** | Recommendation-only by default; simulation, human-approved, and optional Windows Firewall modes |
| **Verifiability** | Model checksums, version registry, SHAP explanations, chained SHA-256 audit log |
| **Self-healing** | Checksum monitoring, quarantine, last-known-good rollback, heartbeat monitoring |

## Quick start (Windows)

```powershell
cd C:\path\to\VCODA_Advanced_Full_Project
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Profile full
.\.venv\Scripts\Activate.ps1
vcoda system-check
```

Place the extracted NF-UQ-NIDS-v2 CSV or Parquet files inside:

```text
data\raw\nf_uq_nids_v2\
```

Then run:

```powershell
vcoda inspect-data --data-dir ".\data\raw\nf_uq_nids_v2"
vcoda prepare-data
vcoda train supervised --task binary
vcoda train supervised --task multiclass
vcoda train anomaly
vcoda train deep --architecture mlp --task binary
vcoda train deep --architecture autoencoder --task binary
vcoda optimise-ensemble
vcoda evaluate
```

Start the API and dashboard in separate terminals:

```powershell
vcoda serve-api
vcoda dashboard
```

| Interface | URL |
|---|---|
| API docs | http://127.0.0.1:8000/docs |
| Dashboard | http://127.0.0.1:8501 |

**Temporal models (CNN, CNN-LSTM, Transformer)** require trustworthy ordering. Dataset
preparation preserves `sequence_group` and `event_time` metadata where the dataset actually
provides them; if reliable sequence order is unavailable, V-CODA refuses temporal training
instead of fabricating one.

## Project structure

```text
src/vcoda/
├── api/            FastAPI service
├── dashboard/       Streamlit operator dashboard
├── data/            inspection, leakage checks, chunked processing
├── models/          supervised + deep training, calibration, registry
├── anomaly/         offline and online anomaly detectors
├── ensemble/        validation-only fusion optimisation
├── mitre/           probable MITRE ATT&CK mapping
├── threat_intel/    IOC/STIX/CTI enrichment
├── drift/           confidence, anomaly, error, attack-rate monitoring
├── response/        deterministic policy engine + reversible actions
├── audit/           append-only chained-hash audit log
├── monitoring/       heartbeats, experiment tracking, self-healing
├── capture/         PCAP and live packet-to-flow aggregation
├── explainability/  SHAP-based attribution
├── reports/         evaluation and analysis report generation
└── cli.py           `vcoda` command-line entry point

scripts/     standalone training/evaluation/monitoring scripts
configs/     YAML configuration (training, inference, response policy, ...)
windows/     PowerShell launch scripts for API, dashboard, watchdog, training
tests/       pytest suite
docs/        full guide set (see below)
```

## Safety and self-healing

`configs/response_policy.yaml` defaults to:

```yaml
mode: recommendation_only
```

Windows Firewall changes are disabled until explicitly configured, confirmed, and executed with
administrator privileges. Model output is never passed to a shell or used to construct arbitrary
commands.

Self-healing is controlled operational recovery, not unrestricted autonomous modification. It:

- detects corrupt active model artefacts and quarantines them;
- rolls back to the last checksum-valid model;
- verifies the audit chain;
- reports stale service heartbeats;
- reverses expired temporary response actions;
- preserves drift windows and recommends retraining.

It **cannot** silently retrain or promote the supervised model. See [SECURITY.md](SECURITY.md)
for the full threat model.

## Documentation

| Guide | Description |
|---|---|
| [Windows setup](docs/WINDOWS_SETUP.md) | Environment and dependency installation |
| [Architecture](docs/ARCHITECTURE.md) | End-to-end data path and module responsibilities |
| [Dataset guide](docs/DATASET_GUIDE.md) | NF-UQ-NIDS-v2 preparation and feature engineering |
| [Training guide](docs/TRAINING_GUIDE.md) | Supervised, deep, and anomaly training |
| [Evaluation guide](docs/EVALUATION_GUIDE.md) | Metrics, calibration, ensemble optimisation |
| [PCAP & live monitoring](docs/PCAP_LIVE_GUIDE.md) | Offline PCAP analysis and live capture |
| [Self-healing](docs/SELF_HEALING.md) | Checksum monitoring and rollback behaviour |
| [API guide](docs/API_GUIDE.md) | FastAPI endpoints and usage |
| [Dashboard guide](docs/DASHBOARD_GUIDE.md) | Streamlit operator dashboard |
| [Configuration](docs/CONFIGURATION.md) | YAML config reference |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues |
| [Lecturer demo](docs/LECTURER_DEMO.md) | Guided demonstration procedure |
| [Viva Q&A](docs/VIVA_QA.md) | Anticipated defence questions |
| [Known limitations](docs/KNOWN_LIMITATIONS.md) | What this project deliberately does not claim |

## Testing

Verified in the packaging environment (see [TEST_REPORT.md](TEST_REPORT.md) for the full report):

- Python syntax compilation across `src`, `scripts`, `dashboard`, `tests`
- CLI help/import and editable install
- **26 automated tests** (leakage checks, audit-chain tamper detection, upload validation,
  MITRE mapping, ensemble fusion, deep-model forward passes, checksum/rollback self-healing,
  API health and validation, response-policy controls)

Not exercised in packaging (require real hardware/data): full NF-UQ-NIDS-v2 training, Windows
Npcap capture, CUDA training, Docker Desktop, and Windows Firewall changes.

```powershell
pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: branch per change, tests for behavioural
changes, `python -m compileall -q src scripts dashboard` and `pytest` before submitting, and
never commit datasets, credentials, or trained model artefacts.

## License

[MIT](LICENSE) © 2026 V-CODA Project. See [ATTRIBUTION.md](ATTRIBUTION.md) for third-party
notices.
