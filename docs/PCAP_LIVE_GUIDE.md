# PCAP and Live Monitoring

## Offline PCAP analysis

After training and promoting a binary model:

```powershell
vcoda analyse-pcap --input ".\data\samples\test_capture.pcap"
```

Custom output:

```powershell
vcoda analyse-pcap --input "C:\captures\traffic.pcapng" --output ".\outputs\traffic_analysis.json"
```

Scapy aggregates packets into bidirectional five-tuple flows and derives bytes, packets, duration, rate, direction ratio, and asymmetry features. Every report contains a feature-compatibility section showing matched, missing, and derived fields.

## Richer extraction

For richer protocol fields, install Wireshark/TShark or run Zeek in WSL/Docker, export a flow table, then use `vcoda predict --input`. The provided core does not hide the mismatch between NF-UQ training features and fields available from a local capture.

## Live monitoring on Windows

1. Install Npcap.
2. Open PowerShell as Administrator only if the chosen interface requires it.
3. Activate the V-CODA environment.
4. Identify interfaces:

```powershell
python -c "from scapy.all import get_if_list; print('\n'.join(get_if_list()))"
```

5. Start capture:

```powershell
vcoda monitor-live --interface "<interface-name>"
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\run_live_monitor.ps1 -Interface "<interface-name>"
```

Stop with `Ctrl+C`. Buffered flows are flushed gracefully to `outputs/live_predictions.jsonl`.

## Reliability warning

A flow model is trustworthy only when capture-time features are compatible with the trained schema. Low feature coverage is included in each prediction explanation. Re-train using a live-compatible common feature subset for operational deployment.
