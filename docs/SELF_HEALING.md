# Controlled Self-Healing

Run one check:

```powershell
vcoda heal
```

Run a watchdog loop:

```powershell
vcoda heal --watch
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\run_watchdog.ps1
```

## Recovery capabilities

- verify all active model SHA-256 checksums;
- quarantine a corrupt active artefact;
- promote the most recent checksum-valid rollback version;
- verify the append-only audit chain;
- detect stale API, inference, or live-monitor heartbeats;
- reverse expired temporary response records;
- preserve drift windows for review;
- continue safe online anomaly updates when policy allows.

## Deliberate limitations

- No silent supervised-model retraining.
- No automatic promotion of a newly trained model.
- No arbitrary process restart by default.
- No destructive remediation.
- No model update using unverified predictions as ground truth.

These restrictions reduce poisoning and unsafe recovery risks. The generated health report is:

```text
artifacts/reports/self_healing_status.json
```
