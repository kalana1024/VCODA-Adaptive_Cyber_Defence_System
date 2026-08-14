# Lecturer Demonstration Procedure

Use real evaluation artefacts generated on your computer. Do not claim results from the controlled sample files.

## 1. Architecture

Open `docs/ARCHITECTURE.md` and explain the independent detection, reasoning, response, audit, and recovery layers.

## 2. Hardware and environment

```powershell
vcoda system-check
```

Show `artifacts/reports/system_check.json`.

## 3. Dataset inspection

```powershell
vcoda inspect-data --data-dir ".\data\raw\nf_uq_nids_v2"
```

Show actual schema, labels, origins, null rates, constants, and fingerprint.

## 4. Leakage controls and preprocessing

```powershell
vcoda prepare-data
```

Show:

```text
artifacts/reports/leakage_report.json
data/processed/manifest.json
```

Explain why IP addresses, labels, and origin are excluded from predictive features.

## 5. Model registry

```powershell
vcoda list-models
```

Show immutable versions, checksums, dataset fingerprint, thresholds, and active models.

## 6. Real evaluation

```powershell
vcoda evaluate
```

Open `artifacts/reports/final_evaluation_report.html`. Discuss macro F1, per-class recall, FPR, PR-AUC, MCC, and latency—not accuracy alone.

## 7. Benign and malicious flow predictions

```powershell
vcoda predict --input ".\data\samples\example_flows.csv"
```

State clearly that the input is a controlled parser/inference test, while the model was trained and evaluated on real held-out NF-UQ data.

## 8. Explainability

Open the stored event explanation under `artifacts/explanations/`. For XGBoost, show SHAP contributions. If the active model exposes only global importance, state that limitation.

## 9. Anomaly detection

Show offline anomaly confidence and the online anomaly model status. Explain that the offline model was trained primarily on benign traffic.

## 10. MITRE mapping

Show a detection with behavioural evidence. Explain that the technique is marked probable and confidence decreases when evidence is classification-only.

## 11. Threat graph

Open the dashboard's MITRE and Threat Graph page. Show alert, host/IP, port, IOC, technique, and incident relationships generated from stored detections.

## 12. Drift

Run a controlled stream or replay and show `artifacts/drift/drift_events.jsonl`. Explain that drift saves a review window and recommends retraining but does not silently promote a model.

## 13. Response recommendation

Show `recommendation_only` mode. Explain thresholds, protected assets, allowlists, corroboration, confirmation, and rollback.

## 14. PCAP

```powershell
vcoda analyse-pcap --input ".\data\samples\test_capture.pcap"
```

Show the feature-compatibility report. Do not claim full reliability when capture features are missing.

## 15. Dashboard and API

```powershell
vcoda serve-api
vcoda dashboard
```

Show API docs, model status, real alerts, model comparison, and audit page.

## 16. Audit verification

```powershell
vcoda verify-audit
```

Explain previous-hash and current-record SHA-256 validation without calling it blockchain.

## 17. Self-healing

```powershell
vcoda heal
```

Explain checksum validation, quarantine, rollback, heartbeat checks, and response expiry cleanup. Emphasise that autonomous supervised retraining is deliberately disabled.
