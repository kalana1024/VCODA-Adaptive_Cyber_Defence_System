# Configuration Guide

## `configs/training.yaml`

Controls source partitions, chunk size, sample caps, preprocessing, model candidates, Optuna trials, neural profiles, sequence length, anomaly false-positive target, and ensemble method.

## `configs/features.yaml`

Contains candidate aliases, leakage exclusions, identifier patterns, and safe derived features. Aliases are used only when the inspected actual schema contains them.

## `configs/inference.yaml`

Controls active-model selection, batch size, schema-coverage threshold, output store, ensemble uncertainty, and documented risk weights.

## `configs/response_policy.yaml`

Controls response mode, protected assets/networks, allowlists, thresholds, action catalogue, Windows Firewall restrictions, and rollback requirements.

## `configs/self_healing.yaml`

Controls checksum checks, rollback, quarantine, heartbeat threshold, safe online updates, and health-report location.

## Profiles

Deep profiles:

- `small`: 8–16 GB RAM, CPU-oriented.
- `medium`: 16–32 GB RAM or modest GPU.
- `high`: 32 GB+ RAM and a capable GPU.

Start small, verify the pipeline, then increase sample sizes and neural capacity.

## Optional threat-intelligence services

V-CODA is local-first and works without external services. To enable optional lookups, copy
`.env.example` to `.env` and set only the credentials you are authorised to use:

```env
ABUSEIPDB_API_KEY=
VIRUSTOTAL_API_KEY=
TAXII_URL=
TAXII_USERNAME=
TAXII_PASSWORD=
```

External lookups are cached, have timeouts, and return structured adapter errors instead of
interrupting inference. MISP JSON exports and STIX indicator bundles can be loaded locally.
Install TAXII support with `pip install -r requirements/cti.txt`.
