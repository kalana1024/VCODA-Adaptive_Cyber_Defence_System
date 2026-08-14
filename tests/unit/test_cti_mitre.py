from pathlib import Path

from vcoda.mitre.reasoner import MitreReasoner
from vcoda.threat_intel.engine import ThreatIntelEngine


def test_cti_extracts_valid_indicators_only():
    engine = ThreatIntelEngine()
    found = {(x["type"], x["value"]) for x in engine.extract("8.8.8.8 bad.example CVE-2025-12345 d41d8cd98f00b204e9800998ecf8427e")}
    assert ("ip", "8.8.8.8") in found
    assert ("domain", "bad.example") in found
    assert ("cve", "CVE-2025-12345") in found


def test_mitre_mapping_reports_probable_not_exact_claim(tmp_path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("""mappings:\n  reconnaissance:\n    tactics: [TA0007]\n    techniques:\n      - id: T1046\n        name: Network Service Discovery\n        confidence: 0.8\n        evidence: [many_destination_ports]\n    mitigations: [M1037]\n""")
    result = MitreReasoner(mapping).map("reconnaissance", {"many_destination_ports": True})
    assert result[0]["mapping_status"] == "probable"
    assert result[0]["technique_id"] == "T1046"


def test_cti_misp_export_and_external_ip_cache(tmp_path, monkeypatch):
    import json
    from vcoda.threat_intel.engine import ThreatIntelEngine

    export = {
        "Event": {
            "Attribute": [
                {"type": "ip-src", "value": "8.8.8.8"},
                {"type": "domain", "value": "malicious.example"},
                {"type": "vulnerability", "value": "CVE-2025-12345"},
            ]
        }
    }
    path = tmp_path / "misp.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    engine = ThreatIntelEngine(cache_seconds=3600)
    assert engine.load_misp_export(path) == 3
    local = engine.enrich(explicit_indicators=[{"type": "domain", "value": "malicious.example"}])
    assert local["reputation"] == 1.0

    calls = {"count": 0}

    def fake_http(url, headers=None, params=None):
        calls["count"] += 1
        if "abuseipdb" in url:
            return {"data": {"abuseConfidenceScore": 80, "totalReports": 10}}
        return {
            "data": {
                "attributes": {
                    "last_analysis_stats": {
                        "malicious": 5,
                        "suspicious": 1,
                        "harmless": 4,
                        "undetected": 0,
                    }
                }
            }
        }

    monkeypatch.setenv("ABUSEIPDB_API_KEY", "test")
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "test")
    monkeypatch.setattr(engine, "_http_json", fake_http)
    first = engine.lookup_ip("8.8.4.4")
    second = engine.lookup_ip("8.8.4.4")
    assert len(first["matches"]) == 2
    assert second == first
    assert calls["count"] == 2


def test_cti_skips_external_lookup_for_private_ip(monkeypatch):
    from vcoda.threat_intel.engine import ThreatIntelEngine

    engine = ThreatIntelEngine()
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "test")
    result = engine.lookup_ip("10.0.0.5")
    assert result == {"matches": [], "errors": [], "services_used": []}
