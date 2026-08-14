from __future__ import annotations

import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vcoda.utils.io import load_json, resolve_path

IP_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
DOMAIN_PATTERN = re.compile(r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}\b")
CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,8}\b", re.IGNORECASE)
HASH_PATTERNS = {
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
}


@dataclass
class CachedResult:
    expires_at: float
    value: dict[str, Any]


class ThreatIntelError(RuntimeError):
    """Raised for recoverable threat-intelligence adapter failures."""


class ThreatIntelEngine:
    """Local-first CTI adapter with optional AbuseIPDB, VirusTotal and TAXII support.

    External services are disabled unless their environment variables are present. Network
    failures are returned as structured adapter errors; they never stop V-CODA inference.
    """

    def __init__(
        self,
        local_ioc_files: list[str | Path] | None = None,
        cache_seconds: int = 3600,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        self.local: dict[tuple[str, str], dict[str, Any]] = {}
        self.cache: dict[tuple[str, str], CachedResult] = {}
        self.cache_seconds = max(0, int(cache_seconds))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        for path in local_ioc_files or []:
            self.load_local(path)

    def extract(self, text: str) -> list[dict[str, str]]:
        indicators: set[tuple[str, str]] = set()
        for candidate in IP_PATTERN.findall(text or ""):
            try:
                ipaddress.ip_address(candidate)
                indicators.add(("ip", candidate))
            except ValueError:
                continue
        for domain in DOMAIN_PATTERN.findall(text or ""):
            if not IP_PATTERN.fullmatch(domain):
                indicators.add(("domain", domain.lower()))
        for cve in CVE_PATTERN.findall(text or ""):
            indicators.add(("cve", cve.upper()))
        for hash_type, pattern in HASH_PATTERNS.items():
            for value in pattern.findall(text or ""):
                indicators.add((hash_type, value.lower()))
        return [{"type": kind, "value": value} for kind, value in sorted(indicators)]

    def load_local(self, path: str | Path) -> int:
        target = resolve_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(target)
        count = 0
        if target.suffix.lower() in {".json", ".stix"}:
            raw = load_json(target, default={})
            count += self._load_json_objects(raw, source_name=target.name)
        else:
            for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
                for extracted in self.extract(line):
                    self.local[(extracted["type"], extracted["value"])] = {
                        "source": target.name,
                        "line": line[:500],
                    }
                    count += 1
        return count

    def load_misp_export(self, path: str | Path) -> int:
        """Load common MISP JSON export shapes without requiring a MISP server."""
        target = resolve_path(path)
        raw = load_json(target, default={})
        events: list[dict[str, Any]] = []
        if isinstance(raw, dict) and isinstance(raw.get("response"), list):
            events = raw["response"]
        elif isinstance(raw, dict) and "Event" in raw:
            events = [raw]
        elif isinstance(raw, list):
            events = raw

        count = 0
        for wrapper in events:
            event = wrapper.get("Event", wrapper) if isinstance(wrapper, dict) else {}
            for attribute in event.get("Attribute", []) or []:
                value = str(attribute.get("value", "")).strip()
                attr_type = str(attribute.get("type", "")).lower()
                mapped = self._misp_type(attr_type, value)
                if mapped:
                    kind, normalized = mapped
                    self.local[(kind, normalized)] = {
                        "source": target.name,
                        "misp_attribute": attribute,
                    }
                    count += 1
        return count

    def fetch_taxii(self, collection_index: int = 0, limit: int = 500) -> dict[str, Any]:
        """Fetch STIX objects from a configured TAXII 2.x server.

        Set TAXII_URL and optionally TAXII_USERNAME/TAXII_PASSWORD. The optional
        ``taxii2-client`` package is imported lazily so V-CODA remains usable offline.
        """
        url = os.getenv("TAXII_URL", "").strip()
        if not url:
            raise ThreatIntelError("TAXII_URL is not configured")
        try:
            from taxii2client.v21 import Server  # type: ignore
        except ImportError as exc:
            raise ThreatIntelError(
                "taxii2-client is not installed; install the optional 'cti' dependencies"
            ) from exc

        username = os.getenv("TAXII_USERNAME") or None
        password = os.getenv("TAXII_PASSWORD") or None
        server = Server(url, user=username, password=password)
        api_roots = list(server.api_roots)
        if not api_roots:
            raise ThreatIntelError("TAXII server exposed no API roots")
        collections = list(api_roots[0].collections)
        if not collections:
            raise ThreatIntelError("TAXII API root exposed no collections")
        if collection_index < 0 or collection_index >= len(collections):
            raise ThreatIntelError(f"TAXII collection index {collection_index} is out of range")

        envelope = collections[collection_index].get_objects(limit=max(1, int(limit)))
        objects = envelope.get("objects", []) if isinstance(envelope, dict) else []
        loaded = self._load_json_objects({"objects": objects}, source_name=f"taxii:{url}")
        return {
            "server": url,
            "collection": getattr(collections[collection_index], "title", None),
            "objects_received": len(objects),
            "indicators_loaded": loaded,
        }

    def enrich(
        self,
        text: str = "",
        explicit_indicators: Iterable[dict[str, str]] | None = None,
        use_external: bool = False,
    ) -> dict[str, Any]:
        indicators = list(explicit_indicators or []) + self.extract(text)
        deduped = {(item["type"], item["value"]): item for item in indicators}
        matches: list[dict[str, Any]] = []
        adapter_errors: list[dict[str, str]] = []
        external_services_used: set[str] = set()

        for key, indicator in deduped.items():
            local = self.local.get(key)
            if local:
                matches.append(
                    {
                        **indicator,
                        "reputation": 1.0,
                        "source": local["source"],
                        "details": local,
                    }
                )
            if use_external and indicator["type"] == "ip":
                external = self.lookup_ip(indicator["value"])
                matches.extend(external["matches"])
                adapter_errors.extend(external["errors"])
                external_services_used.update(external["services_used"])

        reputation = max([float(match.get("reputation", 0.0)) for match in matches], default=0.0)
        return {
            "indicators": list(deduped.values()),
            "matches": matches,
            "reputation": min(max(reputation, 0.0), 1.0),
            "external_services_used": sorted(external_services_used),
            "adapter_errors": adapter_errors,
        }

    def lookup_ip(self, value: str) -> dict[str, Any]:
        """Look up one public IP using configured external services with caching."""
        try:
            parsed = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {value}") from exc
        if parsed.is_private or parsed.is_loopback or parsed.is_link_local or parsed.is_reserved:
            return {"matches": [], "errors": [], "services_used": []}

        cache_key = ("external_ip", value)
        cached = self.cache.get(cache_key)
        if cached and cached.expires_at >= time.time():
            return cached.value

        result: dict[str, Any] = {"matches": [], "errors": [], "services_used": []}
        abuse_key = os.getenv("ABUSEIPDB_API_KEY", "").strip()
        if abuse_key:
            try:
                match = self._lookup_abuseipdb(value, abuse_key)
                if match:
                    result["matches"].append(match)
                result["services_used"].append("abuseipdb")
            except Exception as exc:  # service failure must not stop inference
                result["errors"].append({"service": "abuseipdb", "error": str(exc)[:500]})

        vt_key = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
        if vt_key:
            try:
                match = self._lookup_virustotal(value, vt_key)
                if match:
                    result["matches"].append(match)
                result["services_used"].append("virustotal")
            except Exception as exc:  # service failure must not stop inference
                result["errors"].append({"service": "virustotal", "error": str(exc)[:500]})

        self.cache[cache_key] = CachedResult(
            expires_at=time.time() + self.cache_seconds,
            value=result,
        )
        return result

    def taxii_available(self) -> bool:
        return bool(os.getenv("TAXII_URL"))

    def external_keys_status(self) -> dict[str, bool]:
        return {
            "abuseipdb": bool(os.getenv("ABUSEIPDB_API_KEY")),
            "virustotal": bool(os.getenv("VIRUSTOTAL_API_KEY")),
            "taxii": self.taxii_available(),
        }

    def _lookup_abuseipdb(self, ip_value: str, api_key: str) -> dict[str, Any] | None:
        payload = self._http_json(
            "https://api.abuseipdb.com/api/v2/check",
            headers={"Key": api_key, "Accept": "application/json"},
            params={"ipAddress": ip_value, "maxAgeInDays": 90, "verbose": ""},
        )
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        score = min(max(float(data.get("abuseConfidenceScore", 0.0)) / 100.0, 0.0), 1.0)
        if score <= 0.0 and int(data.get("totalReports", 0) or 0) <= 0:
            return None
        return {
            "type": "ip",
            "value": ip_value,
            "reputation": score,
            "source": "AbuseIPDB",
            "details": {
                "abuse_confidence_score": data.get("abuseConfidenceScore"),
                "total_reports": data.get("totalReports"),
                "country_code": data.get("countryCode"),
                "usage_type": data.get("usageType"),
                "last_reported_at": data.get("lastReportedAt"),
            },
        }

    def _lookup_virustotal(self, ip_value: str, api_key: str) -> dict[str, Any] | None:
        payload = self._http_json(
            f"https://www.virustotal.com/api/v3/ip_addresses/{urllib.parse.quote(ip_value, safe='')}",
            headers={"x-apikey": api_key, "Accept": "application/json"},
        )
        attributes = payload.get("data", {}).get("attributes", {}) if isinstance(payload, dict) else {}
        stats = attributes.get("last_analysis_stats", {}) or {}
        malicious = float(stats.get("malicious", 0) or 0)
        suspicious = float(stats.get("suspicious", 0) or 0)
        total = sum(float(value or 0) for value in stats.values())
        score = (malicious + 0.5 * suspicious) / total if total > 0 else 0.0
        if score <= 0.0:
            return None
        return {
            "type": "ip",
            "value": ip_value,
            "reputation": min(max(score, 0.0), 1.0),
            "source": "VirusTotal",
            "details": {
                "last_analysis_stats": stats,
                "reputation": attributes.get("reputation"),
                "country": attributes.get("country"),
                "network": attributes.get("network"),
            },
        }

    def _http_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "V-CODA/1.0", **(headers or {})},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                if int(response.status) >= 400:
                    raise ThreatIntelError(f"HTTP {response.status}")
                body = response.read(2_000_000)
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            suffix = f"; retry after {retry_after}s" if retry_after else ""
            raise ThreatIntelError(f"HTTP {exc.code}{suffix}") from exc
        except urllib.error.URLError as exc:
            raise ThreatIntelError(f"Network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ThreatIntelError("Threat-intelligence service returned invalid JSON") from exc

    def _load_json_objects(self, raw: Any, source_name: str) -> int:
        objects: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            if isinstance(raw.get("objects"), list):
                objects = raw["objects"]
            elif raw.get("type") == "indicator":
                objects = [raw]
        elif isinstance(raw, list):
            objects = [item for item in raw if isinstance(item, dict)]

        count = 0
        for item in objects:
            if item.get("type") != "indicator":
                continue
            pattern = str(item.get("pattern", ""))
            for extracted in self.extract(pattern):
                self.local[(extracted["type"], extracted["value"])] = {
                    "source": source_name,
                    "stix": item,
                }
                count += 1
        return count

    @staticmethod
    def _misp_type(attr_type: str, value: str) -> tuple[str, str] | None:
        if attr_type in {"ip-src", "ip-dst", "ip-src|port", "ip-dst|port"}:
            ip_value = value.split("|")[0]
            try:
                ipaddress.ip_address(ip_value)
                return "ip", ip_value
            except ValueError:
                return None
        if attr_type in {"domain", "hostname", "domain|ip"}:
            return "domain", value.split("|")[0].lower()
        if attr_type in {"md5", "sha1", "sha256"}:
            return attr_type, value.lower()
        if attr_type == "vulnerability" and CVE_PATTERN.fullmatch(value):
            return "cve", value.upper()
        return None
