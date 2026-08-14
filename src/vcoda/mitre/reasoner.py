from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from vcoda.utils.io import dump_json, load_yaml


CLASS_ALIASES = {
    "reconnaissance": "reconnaissance", "recon": "reconnaissance", "scanning": "reconnaissance",
    "scan": "reconnaissance", "brute force": "brute_force", "brute_force": "brute_force",
    "dos": "dos", "ddos": "dos", "denial of service": "dos",
    "injection": "web_attack", "xss": "web_attack", "web attack": "web_attack", "exploits": "web_attack",
}


class MitreReasoner:
    def __init__(self, mapping_path: str | Path = "configs/mitre_mapping.yaml") -> None:
        self.mapping = load_yaml(mapping_path).get("mappings", {})

    def map(self, attack_category: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
        key = CLASS_ALIASES.get(attack_category.strip().lower(), attack_category.strip().lower().replace(" ", "_"))
        configured = self.mapping.get(key)
        if not configured:
            return []
        observed = {name for name, value in evidence.items() if bool(value)}
        results: list[dict[str, Any]] = []
        for technique in configured.get("techniques", []):
            expected = set(technique.get("evidence", []))
            overlap = observed & expected
            base = float(technique.get("confidence", 0.5))
            confidence = min(0.95, base * (0.5 + 0.5 * len(overlap) / max(len(expected), 1)))
            if not overlap and evidence:
                confidence *= 0.5
            results.append({
                "technique_id": str(technique["id"]),
                "technique_name": str(technique["name"]),
                "tactics": configured.get("tactics", []),
                "confidence": round(confidence, 4),
                "justification": sorted(overlap) or ["classification-only mapping; insufficient behavioural evidence"],
                "mitigations": configured.get("mitigations", []),
                "mapping_status": "probable" if overlap else "low_evidence_candidate",
            })
        return results


class ThreatGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.incidents: dict[str, list[str]] = defaultdict(list)

    def add_detection(self, detection: dict[str, Any]) -> str:
        event_id = str(detection["event_id"])
        timestamp = detection.get("timestamp") or datetime.now(timezone.utc).isoformat()
        self.graph.add_node(event_id, kind="alert", timestamp=timestamp, severity=detection.get("severity"), risk=detection.get("risk_score"))
        for field, kind in [("source_ip", "ip"), ("destination_ip", "ip"), ("destination_port", "port"), ("asset_id", "asset")]:
            value = detection.get(field)
            if value is not None:
                node = f"{kind}:{value}"
                self.graph.add_node(node, kind=kind, value=value)
                relationship = "originated_from" if field == "source_ip" else "targets"
                self.graph.add_edge(event_id, node, relationship=relationship, timestamp=timestamp)
        for mapping in detection.get("mitre", []):
            technique = f"technique:{mapping['technique_id']}"
            self.graph.add_node(technique, kind="mitre_technique", name=mapping["technique_name"])
            self.graph.add_edge(event_id, technique, relationship="supports", confidence=mapping["confidence"])
        for indicator in detection.get("threat_intel", {}).get("indicators", []):
            node = f"ioc:{indicator['type']}:{indicator['value']}"
            self.graph.add_node(node, kind="indicator", value=indicator["value"], indicator_type=indicator["type"])
            self.graph.add_edge(event_id, node, relationship="contains_indicator")
        incident_key = f"{detection.get('source_ip')}|{detection.get('destination_ip')}|{detection.get('predicted_category')}"
        self.incidents[incident_key].append(event_id)
        incident_id = f"incident:{abs(hash(incident_key))}"
        self.graph.add_node(incident_id, kind="incident", key=incident_key)
        self.graph.add_edge(incident_id, event_id, relationship="contains")
        return incident_id

    def correlated_events(self, event_id: str, maximum: int = 50) -> list[str]:
        if event_id not in self.graph:
            return []
        neighbours: set[str] = set()
        for connected in self.graph.successors(event_id):
            for predecessor in self.graph.predecessors(connected):
                if predecessor != event_id and self.graph.nodes[predecessor].get("kind") == "alert":
                    neighbours.add(predecessor)
        return sorted(neighbours)[:maximum]

    def attack_chain(self, incident_id: str) -> list[dict[str, Any]]:
        if incident_id not in self.graph:
            return []
        events = [node for node in self.graph.successors(incident_id) if self.graph.nodes[node].get("kind") == "alert"]
        return sorted(
            [{"event_id": node, **self.graph.nodes[node]} for node in events],
            key=lambda item: str(item.get("timestamp", "")),
        )

    def export(self, path: str | Path = "artifacts/reports/threat_graph.json") -> dict[str, Any]:
        data = nx.node_link_data(self.graph, edges="edges")
        dump_json(data, path)
        return data
