# Security Policy and Responsible Use

## Safe defaults

- Response mode is `recommendation_only`.
- Windows Firewall integration is disabled.
- All disruptive actions require high confidence, high risk, multiple signals, rollback support, and explicit confirmation.
- Model output is never passed to a shell.
- Subprocess calls use fixed argument arrays and `shell=False`.
- API uploads are size-limited, extension-checked, filename-sanitised, and stored in a controlled directory.
- API keys belong in `.env`, never source code.
- Model artefacts are loaded only from the local registry after SHA-256 verification.

## Threat model

Protected assets include model artefacts, preprocessing metadata, the model registry, audit records, response policy, uploaded PCAPs, and external API credentials. Threats include path traversal, malicious uploads, model substitution, dataset poisoning, leakage, arbitrary-command execution, unsafe response, audit tampering, and model drift.

## Reporting vulnerabilities

Do not expose the API to an untrusted network without adding authentication, TLS, a reverse proxy, request throttling, and an organisation-specific access-control policy. Remove sensitive PCAPs and prediction logs before sharing the project folder.

## Windows Firewall warning

Firewall integration is optional and can disrupt connectivity. Test only in an isolated lab, use short expiry periods, maintain console access, and verify rollback. Never automatically block private, loopback, management, or allowlisted addresses.
