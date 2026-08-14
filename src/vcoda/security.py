from __future__ import annotations

import ipaddress
import mimetypes
from pathlib import Path

from vcoda.utils.io import safe_filename

ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".gz", ".parquet", ".pcap", ".pcapng", ".json", ".stix"}
PCAP_EXTENSIONS = {".pcap", ".pcapng"}


def validate_upload_name(name: str, allowed: set[str] | None = None) -> str:
    clean = safe_filename(name)
    suffix = Path(clean).suffix.lower()
    allowed_set = allowed or ALLOWED_UPLOAD_EXTENSIONS
    if suffix not in allowed_set:
        raise ValueError(f"Unsupported file extension: {suffix}")
    return clean


def validate_upload_size(size: int, maximum_bytes: int) -> None:
    if size < 0 or size > maximum_bytes:
        raise ValueError(f"Upload exceeds {maximum_bytes} bytes")


def validate_mime(name: str, mime: str | None) -> None:
    expected, _ = mimetypes.guess_type(name)
    if mime and expected and mime not in {expected, "application/octet-stream"}:
        raise ValueError(f"MIME type {mime!r} does not match {name!r}")


def is_public_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved)


def is_protected_ip(value: str, protected_networks: list[str], allowlisted: list[str]) -> bool:
    ip = ipaddress.ip_address(value)
    if value in allowlisted:
        return True
    return any(ip in ipaddress.ip_network(network) for network in protected_networks)
