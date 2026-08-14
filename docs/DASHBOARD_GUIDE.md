# Dashboard Guide

Start:

```powershell
vcoda dashboard
```

Open `http://127.0.0.1:8501`.

Pages:

- Overview: real processed-flow counts, attack counts, critical alerts, risk timeline.
- Alerts: filters and complete stored detection records.
- MITRE and Threat Graph: nodes and relationships produced by real detections.
- Model Comparison: registry state and held-out evaluation metrics.
- Drift and Self-Healing: drift records and recovery status.
- Audit: chain verification and recent records.
- PCAP and Live Monitoring: generated reports and feature compatibility.
- Settings: read-only YAML view.

The dashboard never inserts static attack totals. Empty data results in an explicit instruction message.
