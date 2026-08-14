# API Guide

Start:

```powershell
vcoda serve-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

- `GET /health`
- `GET /model/status`
- `POST /predict`
- `POST /predict/batch`
- `POST /pcap/analyse`
- `GET /alerts`
- `GET /incidents`
- `GET /drift/status`
- `GET /audit/verify`
- `GET /models`
- `GET /explanations/{event_id}`

Example request:

```json
{
  "source_ip": "203.0.113.15",
  "destination_ip": "198.51.100.30",
  "source_port": 42000,
  "destination_port": 80,
  "protocol": 6,
  "in_bytes": 100000,
  "out_bytes": 100,
  "in_packets": 900,
  "out_packets": 2,
  "flow_duration_ms": 100,
  "tcp_flags": 2,
  "asset_criticality": "high"
}
```

The API is authentication-ready, not Internet-ready. Add TLS, authentication, network access controls, reverse-proxy rate limiting, and organisation-specific authorisation before remote exposure.
