# Known Limitations

1. NF-UQ-NIDS-v2 is a benchmark compilation, not the user's production network.
2. Real operational accuracy cannot be claimed until external and local validation are completed.
3. Full-data deep training can exceed ordinary-PC memory and time; profiles and stratified caps are provided.
4. CNN/CNN-LSTM/Transformer require trustworthy grouping and chronological order.
5. Scapy-derived PCAP features may cover only part of the training schema.
6. Live capture on Windows depends on Npcap, interface permissions, and traffic visibility.
7. Windows Firewall integration was not executed during packaging and remains disabled by default.
8. External CTI services require user-supplied keys and can be stale, rate-limited, or inaccurate.
9. SHAP is faithful for supported models but can be computationally expensive.
10. Anomaly detection identifies deviation, not malicious intent or guaranteed zero-day attacks.
11. Drift detection can identify statistical change but not its cause.
12. The API requires additional authentication/TLS controls before network exposure.
13. Self-healing cannot recover when no healthy registered model exists.
14. The project is a production-style research platform, not a certified commercial IDS/IPS.

- AbuseIPDB, VirusTotal and TAXII adapters require user-supplied credentials or server details,
  network access, and compliance with the provider's terms and rate limits. They were unit-tested
  with mocked service responses, not against live accounts.
