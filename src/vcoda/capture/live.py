from __future__ import annotations

import importlib.util
import signal
import threading
import time
from pathlib import Path
from typing import Any

from vcoda.capture.flow import FlowState, packet_to_flow
from vcoda.engine import VCODAEngine
from vcoda.monitoring.heartbeat import write_heartbeat
from vcoda.utils.io import append_jsonl


def monitor_live(
    interface: str,
    *,
    flush_seconds: int = 10,
    idle_timeout_seconds: int = 30,
    output: str | Path = "outputs/live_predictions.jsonl",
) -> None:
    if importlib.util.find_spec("scapy") is None:
        raise RuntimeError("Live monitoring requires Scapy and Windows Npcap")
    # See capture/pcap.py: Ether/IP/TCP/UDP dissectors must be registered (by
    # importing these modules) before any packet is captured/dissected, or every
    # packet silently degrades to unparseable Raw bytes.
    import scapy.layers.inet  # noqa: F401
    import scapy.layers.l2  # noqa: F401
    from scapy.sendrecv import AsyncSniffer
    engine = VCODAEngine()
    flows: dict[tuple[Any, ...], FlowState] = {}
    reverse_lookup: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    lock = threading.Lock()
    stop = threading.Event()

    def handle(packet: Any) -> None:
        converted = packet_to_flow(packet)
        if converted is None:
            return
        key, _, values = converted
        reverse = values["reverse_key"]
        with lock:
            canonical = reverse_lookup.get(key, key)
            forward = canonical == key
            if canonical not in flows:
                flows[canonical] = FlowState(*key, start_time=values["timestamp"], end_time=values["timestamp"])
                reverse_lookup[reverse] = canonical
            flows[canonical].update(values["timestamp"], values["size"], forward, values["flags"])

    def flush(force: bool = False) -> None:
        now = time.time()
        completed: list[tuple[Any, ...]] = []
        with lock:
            for key, flow in flows.items():
                if force or now - flow.end_time >= idle_timeout_seconds:
                    completed.append(key)
            records = [flows.pop(key).to_record() for key in completed]
        for record in records:
            result = engine.predict(record, explain=False)
            append_jsonl(output, result)
        write_heartbeat("live_monitor", {"interface": interface, "active_flows": len(flows), "flushed": len(records)})

    def request_stop(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    sniffer = AsyncSniffer(iface=interface, prn=handle, store=False)
    sniffer.start()
    try:
        while not stop.wait(flush_seconds):
            flush()
    finally:
        sniffer.stop()
        flush(force=True)
