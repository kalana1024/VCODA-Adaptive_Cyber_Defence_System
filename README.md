<div align="center">

<!-- BANNER -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1f2e,100:0f3460&height=220&section=header&text=V-CODA&fontSize=80&fontColor=58a6ff&fontAlignY=38&desc=Verifiable%20Cybersecurity-Oriented%20Detection%20%26%20Adaptive%20Response%20Architecture&descAlignY=58&descSize=16&descColor=8b949e&animation=fadeIn" />
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:e8f4fd,50:c3e0f5,100:a8d5f0&height=220&section=header&text=V-CODA&fontSize=80&fontColor=0550ae&fontAlignY=38&desc=Verifiable%20Cybersecurity-Oriented%20Detection%20%26%20Adaptive%20Response%20Architecture&descAlignY=58&descSize=16&descColor=57606a&animation=fadeIn" alt="V-CODA Banner" />
</picture>

<br/>

<!-- BADGES ROW 1 -->
[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Status](https://img.shields.io/badge/Status-Research%20Grade-f97316?style=for-the-badge&logo=academia&logoColor=white)](#)

<br/>

<!-- BADGES ROW 2 -->
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-189AB4?style=flat-square&logo=xgboost&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MITRE ATT&CK](https://img.shields.io/badge/Framework-MITRE%20ATT%26CK-e11d48?style=flat-square&logo=shield&logoColor=white)](https://attack.mitre.org/)
[![Tests](https://img.shields.io/badge/Tests-26%20Passed-4ade80?style=flat-square&logo=pytest&logoColor=white)](TEST_REPORT.md)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#)

<br/><br/>

> **V-CODA** is a research-grade, production-style **hybrid network intrusion detection and adaptive response platform**.
> It combines gradient-boosted classifiers, deep sequence models, unsupervised anomaly detection,
> validation-trained ensemble fusion, MITRE ATT&CK reasoning, and an audit-chained self-healing layer —
> with **no falsely pretrained model** and **no bundled dataset**.
> You train it on your own data and it tells you honestly what it can and cannot do.

<br/>

[![Built for NF-UQ-NIDS-v2](https://img.shields.io/badge/Built%20for-NF--UQ--NIDS--v2-7c3aed?style=for-the-badge&logo=database&logoColor=white)](https://research.unsw.edu.au/projects/nf-uq-nids-v2)
&nbsp;
[![No Pretrained Weights](https://img.shields.io/badge/No%20Pretrained%20Weights-Honest%20by%20Design-dc2626?style=for-the-badge&logo=scales&logoColor=white)](#)

</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to expand</b></summary>

| # | Section |
|---|---------|
| 1 | [🎯 Why V-CODA](#-why-v-coda) |
| 2 | [🏗️ Detection Architecture](#️-detection-architecture) |
| 3 | [⚡ Quick Start (Windows)](#-quick-start-windows) |
| 4 | [📁 Project Structure](#-project-structure) |
| 5 | [🛡️ Safety & Self-Healing](#️-safety--self-healing) |
| 6 | [📚 Documentation](#-documentation) |
| 7 | [🧪 Testing](#-testing) |
| 8 | [🤝 Contributing](#-contributing) |
| 9 | [⚖️ License](#️-license) |

</details>

---

## 🎯 Why V-CODA

> Most academic NIDS prototypes stop at *"trained a classifier, got an F1 score."*
> V-CODA is built around the parts that make a detector **usable in a real operational loop**.

<br/>

<table>
<tr>
<td width="50%">

### 🧠 Intelligence
- **Multiple learners, one decision** — tree ensembles, deep sequence models, and unsupervised anomaly detectors fused with a validation-trained combiner, not hand-picked thresholds
- **Explainable and traceable** — SHAP attributions where supported, probable MITRE ATT&CK mapping, and an append-only SHA-256 audit chain for every decision and response action

</td>
<td width="50%">

### 🔄 Adaptability
- **Adaptive, not reckless** — drift is monitored (ADWIN / Page-Hinkley) and anomaly models can update online; the supervised model never silently retrains or self-promotes
- **Safe by default** — response actions are recommendation-only unless explicitly configured, confirmed, and run with administrator privileges

</td>
</tr>
<tr>
<td colspan="2">

### 🔧 Reliability
- **Self-healing** — corrupted model artefacts are quarantined and rolled back to the last checksum-valid version automatically

</td>
</tr>
</table>

---

## 🏗️ Detection Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│              INPUT  ─  CSV / Parquet / PCAP / Live Packets / CTI        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Schema & Security Validation│
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Feature Compatibility &     │
                    │  Preprocessing               │
                    └──┬────────┬────────┬────────┘
                       │        │        │        │
            ┌──────────▼──┐ ┌───▼───┐ ┌──▼────┐ ┌▼──────────┐
            │  XGBoost    │ │ Deep  │ │Anomaly│ │ CTI/Rules │
            │  binary/    │ │ MLP   │ │ ISO F │ │ IOC/STIX  │
            │  multiclass │ │ CNN   │ │ AE    │ │           │
            │  + LightGBM │ │ LSTM  │ │ HST   │ │           │
            │  + CatBoost │ │ Trans.│ │       │ │           │
            └──────────┬──┘ └───┬───┘ └──┬────┘ └┬──────────┘
                       └────────┴────────┴────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Validation-Trained Ensemble  │
                    │ Fusion (Stacking / Voting)   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Uncertainty + Disagreement + │
                    │     Risk Calculation         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Probable MITRE ATT&CK       │
                    │      Reasoning               │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Incident / Threat Graph    │
                    │        Correlation           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Deterministic Response-     │
                    │  Policy Verification         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Recommendation / Simulation  │
                    │     / Approved Action        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Audit Chain + Registry +    │
                    │      Drift Monitor           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Checksum / Rollback          │
                    │   Self-Healing Manager       │
                    └─────────────────────────────┘
```

</div>

<br/>

### Layer Breakdown

| Layer | Components |
|:------|:-----------|
| 🌲 **Known-Attack Models** | XGBoost *(primary)*, LightGBM, CatBoost, Random Forest, Extra Trees |
| 🧠 **Deep Models** | PyTorch MLP, 1D-CNN, CNN-LSTM, Transformer encoder, benign-only autoencoder |
| 🔍 **Anomaly Models** | Isolation Forest, optional LOF / One-Class SVM, autoencoder, River Half-Space Trees |
| 🔗 **Fusion** | Validation-trained logistic stacking or validation-optimised weighted soft voting |
| 🎯 **Threat Reasoning** | Probable MITRE ATT&CK mapping, IOC/STIX enrichment, NetworkX incident graph |
| 📈 **Adaptation** | ADWIN / Page-Hinkley drift monitoring, safe online anomaly updates |
| ⚡ **Response** | Recommendation-only by default; simulation, human-approved, and optional Windows Firewall modes |
| ✅ **Verifiability** | Model checksums, version registry, SHAP explanations, chained SHA-256 audit log |
| 🔧 **Self-Healing** | Checksum monitoring, quarantine, last-known-good rollback, heartbeat monitoring |

---

## ⚡ Quick Start (Windows)

### 1️⃣  Setup

```powershell
cd C:\path\to\VCODA_Advanced_Full_Project
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Profile full
.\.venv\Scripts\Activate.ps1
vcoda system-check
```

### 2️⃣  Prepare Data

Place the extracted NF-UQ-NIDS-v2 CSV or Parquet files inside:

```
data\raw\nf_uq_nids_v2\
```

### 3️⃣  Train

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

### 4️⃣  Run

Start the API and dashboard in separate terminals:

```powershell
# Terminal 1 — API Server
vcoda serve-api

# Terminal 2 — Operator Dashboard
vcoda dashboard
```

<br/>

<div align="center">

| 🌐 Interface | 🔗 URL |
|:---:|:---:|
| **Swagger API Docs** | [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) |
| **Streamlit Dashboard** | [`http://127.0.0.1:8501`](http://127.0.0.1:8501) |

</div>

<br/>

> [!IMPORTANT]
> **Temporal models** (CNN, CNN-LSTM, Transformer) require trustworthy ordering. Dataset preparation preserves `sequence_group` and `event_time` metadata where the dataset actually provides them; if reliable sequence order is unavailable, V-CODA **refuses** temporal training instead of fabricating one.

---

## 📁 Project Structure

<details>
<summary><b>Click to expand full project tree</b></summary>

```
VCODA_Advanced_Full_Project/
│
├── 📦 src/vcoda/                   ← Core Python package
│   ├── api/                        ·  FastAPI service
│   ├── dashboard/                  ·  Streamlit operator dashboard
│   ├── data/                       ·  Inspection, leakage checks, chunked processing
│   ├── models/                     ·  Supervised + deep training, calibration, registry
│   ├── anomaly/                    ·  Offline and online anomaly detectors
│   ├── ensemble/                   ·  Validation-only fusion optimisation
│   ├── mitre/                      ·  Probable MITRE ATT&CK mapping
│   ├── threat_intel/               ·  IOC/STIX/CTI enrichment
│   ├── drift/                      ·  Confidence, anomaly, error, attack-rate monitoring
│   ├── response/                   ·  Deterministic policy engine + reversible actions
│   ├── audit/                      ·  Append-only chained-hash audit log
│   ├── monitoring/                 ·  Heartbeats, experiment tracking, self-healing
│   ├── capture/                    ·  PCAP and live packet-to-flow aggregation
│   ├── explainability/             ·  SHAP-based attribution
│   ├── reports/                    ·  Evaluation and analysis report generation
│   └── cli.py                      ·  `vcoda` command-line entry point
│
├── 📜 scripts/                     ← Standalone training/evaluation/monitoring scripts
├── ⚙️  configs/                    ← YAML configuration (training, inference, response policy)
├── 🪟 windows/                    ← PowerShell launch scripts
├── 🧪 tests/                       ← pytest suite
└── 📚 docs/                        ← Full guide set
```

</details>

---

## 🛡️ Safety & Self-Healing

### Response Policy

`configs/response_policy.yaml` defaults to:

```yaml
mode: recommendation_only
```

> [!WARNING]
> Windows Firewall changes are **disabled** until explicitly configured, confirmed, and executed with **administrator privileges**. Model output is never passed to a shell or used to construct arbitrary commands.

### Self-Healing Capabilities

Self-healing is controlled operational recovery, **not** unrestricted autonomous modification:

```
┌─────────────────────────────────────────────┐
│              SELF-HEALING MANAGER           │
├─────────────────────────────────────────────┤
│  ✅  Detects & quarantines corrupt artefacts │
│  🔄  Rolls back to last checksum-valid model │
│  🔗  Verifies the full audit chain          │
│  💓  Reports stale service heartbeats        │
│  ↩️  Reverses expired temporary actions      │
│  📊  Preserves drift windows                │
│  ⚠️  Recommends retraining when needed      │
├─────────────────────────────────────────────┤
│  ❌  CANNOT silently retrain supervised model│
│  ❌  CANNOT self-promote model versions      │
└─────────────────────────────────────────────┘
```

See [SECURITY.md](SECURITY.md) for the full threat model.

---

## 📚 Documentation

<div align="center">

| 📄 Guide | 📝 Description |
|:---------|:---------------|
| [🪟 Windows Setup](docs/WINDOWS_SETUP.md) | Environment and dependency installation |
| [🏗️ Architecture](docs/ARCHITECTURE.md) | End-to-end data path and module responsibilities |
| [📊 Dataset Guide](docs/DATASET_GUIDE.md) | NF-UQ-NIDS-v2 preparation and feature engineering |
| [🎓 Training Guide](docs/TRAINING_GUIDE.md) | Supervised, deep, and anomaly training |
| [📈 Evaluation Guide](docs/EVALUATION_GUIDE.md) | Metrics, calibration, ensemble optimisation |
| [📡 PCAP & Live Monitoring](docs/PCAP_LIVE_GUIDE.md) | Offline PCAP analysis and live capture |
| [🔧 Self-Healing](docs/SELF_HEALING.md) | Checksum monitoring and rollback behaviour |
| [🌐 API Guide](docs/API_GUIDE.md) | FastAPI endpoints and usage |
| [📺 Dashboard Guide](docs/DASHBOARD_GUIDE.md) | Streamlit operator dashboard |
| [⚙️ Configuration](docs/CONFIGURATION.md) | YAML config reference |
| [🔍 Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and resolutions |
| [🎬 Lecturer Demo](docs/LECTURER_DEMO.md) | Guided demonstration procedure |
| [💬 Viva Q&A](docs/VIVA_QA.md) | Anticipated defence questions and answers |
| [⚠️ Known Limitations](docs/KNOWN_LIMITATIONS.md) | What this project deliberately does not claim |

</div>

---

## 🧪 Testing

<div align="center">

[![Tests Passed](https://img.shields.io/badge/Automated%20Tests-26%20Passed-4ade80?style=for-the-badge&logo=pytest&logoColor=white)](TEST_REPORT.md)
[![Syntax](https://img.shields.io/badge/Syntax%20Compilation-Passed-4ade80?style=for-the-badge&logo=python&logoColor=white)](TEST_REPORT.md)
[![CLI](https://img.shields.io/badge/CLI%20Import%20Smoke-Passed-4ade80?style=for-the-badge&logo=windowsterminal&logoColor=white)](TEST_REPORT.md)

</div>

<br/>

Verified in the packaging environment — see [TEST_REPORT.md](TEST_REPORT.md) for the full report:

- ✅ Python syntax compilation across `src`, `scripts`, `dashboard`, `tests`
- ✅ CLI help/import and editable install
- ✅ **26 automated tests** covering:

<details>
<summary>View test coverage areas</summary>

| Category | Tests |
|:---------|:------|
| Data Integrity | Leakage checks, upload validation |
| Security | Audit-chain tamper detection |
| Intelligence | MITRE mapping, ensemble fusion |
| Models | Deep-model forward passes, checksum/rollback self-healing |
| API | Health and validation endpoints |
| Policy | Response-policy controls |

</details>

> [!NOTE]
> Not exercised in packaging (require real hardware/data): full NF-UQ-NIDS-v2 training, Windows Npcap capture, CUDA training, Docker Desktop, and Windows Firewall changes.

### Run Tests

```powershell
pytest
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting.

**In short:**

```
✅  Branch per change
✅  Tests for behavioural changes
✅  python -m compileall -q src scripts dashboard
✅  pytest must pass before submitting

❌  Never commit datasets
❌  Never commit credentials
❌  Never commit trained model artefacts
```

---

## ⚖️ License

<div align="center">

This project is licensed under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

[MIT](LICENSE) © 2026 V-CODA Project

See [ATTRIBUTION.md](ATTRIBUTION.md) for third-party notices.

</div>

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1f2e,100:0f3460&height=120&section=footer" />
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:e8f4fd,50:c3e0f5,100:a8d5f0&height=120&section=footer" alt="footer" />
</picture>

<sub>Built with ❤️ for honest, verifiable, and adaptive cybersecurity research.</sub>

</div>
