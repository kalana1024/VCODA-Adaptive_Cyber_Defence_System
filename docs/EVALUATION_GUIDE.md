# Evaluation Guide

Run:

```powershell
vcoda evaluate
```

Outputs:

```text
artifacts/reports/final_evaluation_report.json
artifacts/reports/final_evaluation_report.html
```

Metrics include, where applicable:

- macro precision, recall, and F1;
- weighted F1;
- balanced accuracy;
- Matthews correlation coefficient;
- false-positive and false-negative rates;
- ROC-AUC;
- PR-AUC;
- Brier score;
- confusion matrix and per-class report;
- inference latency and throughput from training reports.

## Correct interpretation

- Use the held-out test partition only once for final claims.
- Do not tune thresholds after inspecting final test results.
- Report per-class recall because rare attacks may be missed despite high overall accuracy.
- Keep formal classifier evaluation separate from end-to-end response demonstrations.
- Treat cross-source results as evidence of generalisation, not a guarantee for real enterprise networks.

No values are pre-populated. The dashboard remains empty until real reports are generated.
