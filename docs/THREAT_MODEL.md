# Threat Model

## Assets

- raw and processed network data;
- model artefacts and preprocessors;
- registry metadata and checksums;
- audit and response ledgers;
- API keys and environment configuration;
- uploaded PCAP files;
- incident and explanation records.

## Threats and controls

| Threat | Control |
|---|---|
| Dataset leakage | schema inspection, explicit exclusions, source/group split |
| Poisoning | benign-only anomaly training, no silent supervised updates |
| Model replacement | registry SHA-256 verification and rollback |
| Path traversal | basename sanitisation and controlled directories |
| Malicious upload | extension/MIME/size checks and no shell execution |
| Arbitrary command execution | fixed response catalogue and `shell=False` |
| Unsafe blocking | recommendation-only default, allowlists, protected assets, confirmation, rollback |
| Audit modification | chained hashes and verification command |
| Credential exposure | `.env`, redaction guidance, no hard-coded keys |
| Denial of service | batch/upload limits, bounded capture buffers, configurable sample caps |
| False confidence | feature-compatibility report and uncertainty/disagreement output |
