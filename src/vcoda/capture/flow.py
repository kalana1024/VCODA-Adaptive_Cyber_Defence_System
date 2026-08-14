from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowState:
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: int
    start_time: float
    end_time: float
    in_bytes: int = 0
    out_bytes: int = 0
    in_packets: int = 0
    out_packets: int = 0
    tcp_flags: int = 0

    def update(self, timestamp: float, size: int, forward: bool, flags: int = 0) -> None:
        self.end_time = max(self.end_time, timestamp)
        if forward:
            self.in_bytes += size
            self.in_packets += 1
        else:
            self.out_bytes += size
            self.out_packets += 1
        self.tcp_flags |= flags

    def to_record(self) -> dict[str, Any]:
        duration_ms = max((self.end_time - self.start_time) * 1000.0, 0.0)
        total_bytes = self.in_bytes + self.out_bytes
        total_packets = self.in_packets + self.out_packets
        seconds = max(duration_ms / 1000.0, 1e-6)
        return {
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
            "in_bytes": self.in_bytes,
            "out_bytes": self.out_bytes,
            "in_packets": self.in_packets,
            "out_packets": self.out_packets,
            "flow_duration_ms": duration_ms,
            "tcp_flags": self.tcp_flags,
            "total_bytes": total_bytes,
            "total_packets": total_packets,
            "bytes_per_packet": total_bytes / max(total_packets, 1),
            "packets_per_second": total_packets / seconds,
            "bytes_per_second": total_bytes / seconds,
            "direction_byte_ratio": self.in_bytes / max(self.out_bytes, 1),
            "direction_packet_ratio": self.in_packets / max(self.out_packets, 1),
            "packet_asymmetry": abs(self.in_packets - self.out_packets),
            "byte_asymmetry": abs(self.in_bytes - self.out_bytes),
        }


def packet_to_flow(packet: Any) -> tuple[tuple[Any, ...], bool, dict[str, Any]] | None:
    try:
        from scapy.layers.inet import IP, TCP, UDP
        if IP not in packet:
            return None
        source_ip, destination_ip = packet[IP].src, packet[IP].dst
        protocol = int(packet[IP].proto)
        source_port = destination_port = 0
        flags = 0
        if TCP in packet:
            source_port, destination_port = int(packet[TCP].sport), int(packet[TCP].dport)
            flags = int(packet[TCP].flags)
        elif UDP in packet:
            source_port, destination_port = int(packet[UDP].sport), int(packet[UDP].dport)
        forward_key = (source_ip, destination_ip, source_port, destination_port, protocol)
        reverse_key = (destination_ip, source_ip, destination_port, source_port, protocol)
        return forward_key, True, {"reverse_key": reverse_key, "timestamp": float(packet.time), "size": len(packet), "flags": flags}
    except Exception:
        return None
