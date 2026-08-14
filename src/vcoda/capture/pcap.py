from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from vcoda.capture.flow import FlowState, packet_to_flow
from vcoda.engine import VCODAEngine
from vcoda.security import PCAP_EXTENSIONS, validate_upload_name
from vcoda.utils.io import dump_json, resolve_path


def extract_pcap_flows(path: str | Path, maximum_packets: int | None = None) -> pd.DataFrame:
    if importlib.util.find_spec("scapy") is None:
        raise RuntimeError("PCAP analysis requires Scapy. Install requirements/development.txt")
    # Ether/IP/TCP/UDP dissectors self-register via bind_layers() as a side effect
    # of importing these modules. rdpcap dissects every packet immediately as it
    # reads the file, so these must be imported *before* rdpcap runs — importing
    # them lazily inside packet_to_flow() (called afterwards, per packet) is too
    # late and silently degrades every packet to unparseable Raw bytes, which is
    # why real captures were producing zero flows.
    import scapy.layers.inet  # noqa: F401
    import scapy.layers.l2  # noqa: F401
    from scapy.utils import rdpcap
    target = resolve_path(path)
    validate_upload_name(target.name, PCAP_EXTENSIONS)
    flows: dict[tuple[Any, ...], FlowState] = {}
    reverse_lookup: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    packets = rdpcap(str(target))
    for index, packet in enumerate(packets):
        if maximum_packets is not None and index >= maximum_packets:
            break
        converted = packet_to_flow(packet)
        if converted is None:
            continue
        key, _, values = converted
        reverse = values["reverse_key"]
        canonical = reverse_lookup.get(key, key)
        forward = canonical == key
        if canonical not in flows:
            flows[canonical] = FlowState(*key, start_time=values["timestamp"], end_time=values["timestamp"])
            reverse_lookup[reverse] = canonical
        flows[canonical].update(values["timestamp"], values["size"], forward, values["flags"])
    return pd.DataFrame([flow.to_record() for flow in flows.values()])


def feature_compatibility(engine: VCODAEngine, frame: pd.DataFrame) -> dict[str, Any]:
    if engine.supervised_binary is None or engine.supervised_binary.preprocessor.schema is None:
        return {"available_live_features": frame.columns.tolist(), "matched": [], "missing": [], "coverage": 0.0}
    schema = engine.supervised_binary.preprocessor.schema.all_features
    matched = [column for column in schema if column in frame.columns]
    missing = [column for column in schema if column not in frame.columns]
    return {
        "training_features": schema, "available_live_features": frame.columns.tolist(),
        "matched": matched, "missing": missing, "derived": [column for column in frame.columns if column not in {"source_ip", "destination_ip", "source_port", "destination_port", "protocol", "in_bytes", "out_bytes", "in_packets", "out_packets", "flow_duration_ms", "tcp_flags"}],
        "coverage": len(matched) / max(len(schema), 1),
        "reliability_warning": "Predictions are less reliable when live feature coverage is below the trained schema coverage requirement.",
    }


def analyse_pcap(path: str | Path, output: str | Path = "outputs/pcap_analysis.json", maximum_packets: int | None = None) -> dict[str, Any]:
    engine = VCODAEngine()
    flows = extract_pcap_flows(path, maximum_packets=maximum_packets)
    results = [engine.predict(record, explain=False) for record in flows.to_dict(orient="records")]
    report = {"pcap": str(resolve_path(path)), "flow_count": len(flows), "feature_compatibility": feature_compatibility(engine, flows), "results": results}
    dump_json(report, output)
    return report
