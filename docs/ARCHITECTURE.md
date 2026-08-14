# V-CODA Architecture

## End-to-end data path

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

## Separation of responsibilities

- `data`: inspection, leakage checks, chunked processing, partitions.
- `models`: supervised and deep training, calibration, metrics, registry.
- `anomaly`: offline and online anomaly models.
- `ensemble`: validation-only fusion optimisation.
- `mitre` and `threat_intel`: evidence enrichment and probable mappings.
- `drift`: confidence, anomaly, error, and attack-rate change monitoring.
- `response`: deterministic policy and reversible actions.
- `audit`: append-only chained hashes.
- `monitoring`: heartbeats, experiment tracking, and self-healing.
- `capture`: PCAP and live packet-to-flow aggregation.
- `api`, `dashboard`, `cli`: operational interfaces.

## Model promotion

Training produces a staging artefact. Registration copies it into an immutable version directory and records its checksum, dataset fingerprint, feature list, preprocessing version, metrics, threshold, environment, and hyperparameters. Promotion updates `models/registry/active.json`. Existing active records move to rollback history.

## No test leakage

Preprocessing is fitted on training data. Calibration and ensemble weights are fitted on validation data. Final metrics are computed on the held-out test partition. Source IP, destination IP, direct labels, dataset origin, and detected high-cardinality identifiers are excluded from predictive features by default.
