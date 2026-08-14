from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from vcoda.audit.chain import AuditChain
from vcoda.capture.pcap import analyse_pcap
from vcoda.engine import VCODAEngine
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.heartbeat import write_heartbeat
from vcoda.security import PCAP_EXTENSIONS, validate_mime, validate_upload_name, validate_upload_size
from vcoda.utils.io import iter_jsonl, load_json, resolve_path, safe_filename
from vcoda.utils.logging import configure_logging, set_request_id
from vcoda.utils.system import system_check

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/metrics"}
RATE_LIMITED_PATHS = {"/predict", "/predict/batch", "/pcap/analyse"}


class _Metrics:
    """Hand-rolled Prometheus text-exposition counters (no prometheus_client
    dependency installed): request counts by path/status, and prediction outcomes
    by ensemble_status, so /metrics can back a Grafana panel without new deps."""

    def __init__(self) -> None:
        self.requests_total: dict[tuple[str, int], int] = defaultdict(int)
        self.predictions_total: dict[str, int] = defaultdict(int)
        self.request_latency_seconds: list[float] = []

    def observe_request(self, path: str, status_code: int, duration: float) -> None:
        self.requests_total[(path, status_code)] += 1
        self.request_latency_seconds.append(duration)
        if len(self.request_latency_seconds) > 2000:
            self.request_latency_seconds = self.request_latency_seconds[-2000:]

    def observe_prediction(self, ensemble_status: str) -> None:
        self.predictions_total[ensemble_status] += 1

    def render(self) -> str:
        lines = ["# HELP vcoda_requests_total Total HTTP requests", "# TYPE vcoda_requests_total counter"]
        for (path, status_code), count in self.requests_total.items():
            lines.append(f'vcoda_requests_total{{path="{path}",status="{status_code}"}} {count}')
        lines += ["# HELP vcoda_predictions_total Predictions by ensemble status", "# TYPE vcoda_predictions_total counter"]
        for status, count in self.predictions_total.items():
            lines.append(f'vcoda_predictions_total{{ensemble_status="{status}"}} {count}')
        latencies = self.request_latency_seconds
        if latencies:
            lines += ["# HELP vcoda_request_latency_seconds_avg Average request latency", "# TYPE vcoda_request_latency_seconds_avg gauge"]
            lines.append(f"vcoda_request_latency_seconds_avg {sum(latencies) / len(latencies):.6f}")
        return "\n".join(lines) + "\n"


class _RateLimiter:
    """Fixed-window-free sliding counter per client IP, in-memory only. Disabled
    (unlimited) unless VCODA_RATE_LIMIT_PER_MINUTE is set, so default local/dev
    usage and the dashboard are unaffected."""

    def __init__(self, limit_per_minute: int | None) -> None:
        self.limit = limit_per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_key: str) -> bool:
        if not self.limit:
            return True
        now = time.time()
        window = self.hits[client_key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(now)
        return True


metrics = _Metrics()
rate_limiter = _RateLimiter(int(os.environ["VCODA_RATE_LIMIT_PER_MINUTE"]) if os.environ.get("VCODA_RATE_LIMIT_PER_MINUTE") else None)


class FlowRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: str | None = None
    source_ip: str | None = None
    destination_ip: str | None = None
    source_port: int | None = Field(default=None, ge=0, le=65535)
    destination_port: int | None = Field(default=None, ge=0, le=65535)
    asset_criticality: str = "medium"


class BatchRequest(BaseModel):
    flows: list[FlowRequest]


@lru_cache(maxsize=1)
def get_engine() -> VCODAEngine:
    return VCODAEngine()


def create_app() -> FastAPI:
    configure_logging()
    api = FastAPI(title="V-CODA API", version="1.0.0", docs_url="/docs")

    @api.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = set_request_id(request.headers.get("X-Request-ID"))
        started = time.perf_counter()
        path = request.url.path
        api_key = os.environ.get("VCODA_API_KEY")
        if api_key and path not in OPEN_PATHS and request.headers.get("X-API-Key") != api_key:
            response = JSONResponse(status_code=401, content={"error": "unauthorized", "request_id": request_id})
            response.headers["X-Request-ID"] = request_id
            metrics.observe_request(path, 401, time.perf_counter() - started)
            return response
        if path in RATE_LIMITED_PATHS and not rate_limiter.check(request.client.host if request.client else "unknown"):
            response = JSONResponse(status_code=429, content={"error": "rate_limited", "request_id": request_id})
            response.headers["X-Request-ID"] = request_id
            metrics.observe_request(path, 429, time.perf_counter() - started)
            return response
        try:
            response = await call_next(request)
        except Exception as exc:
            response = JSONResponse(
                status_code=500,
                content={"error": "internal_error", "message": str(exc), "request_id": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        write_heartbeat("api", {"path": path})
        metrics.observe_request(path, response.status_code, time.perf_counter() - started)
        return response

    @api.get("/metrics")
    def metrics_endpoint() -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "system": system_check(),
            "model_registry_present": Path("models/registry").exists(),
        }

    @api.get("/model/status")
    def model_status() -> dict[str, Any]:
        return get_engine().status()

    @api.post("/predict")
    def predict(flow: FlowRequest) -> dict[str, Any]:
        try:
            result = get_engine().predict(flow.model_dump(exclude_none=True), explain=True)
            metrics.observe_prediction(str(result.get("ensemble_status", "unknown")))
            return result
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.post("/predict/batch")
    def predict_batch(batch: BatchRequest) -> dict[str, Any]:
        if len(batch.flows) > MAX_BATCH_ROWS:
            raise HTTPException(status_code=413, detail=f"Maximum batch size is {MAX_BATCH_ROWS}")
        engine = get_engine()
        results = [engine.predict(flow.model_dump(exclude_none=True), explain=False) for flow in batch.flows]
        for result in results:
            metrics.observe_prediction(str(result.get("ensemble_status", "unknown")))
        return {"count": len(batch.flows), "results": results}

    @api.post("/pcap/analyse")
    async def pcap_analyse(file: UploadFile = File(...)) -> dict[str, Any]:
        clean = validate_upload_name(file.filename or "capture.pcap", PCAP_EXTENSIONS)
        validate_mime(clean, file.content_type)
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        validate_upload_size(len(content), MAX_UPLOAD_BYTES)
        upload_dir = resolve_path("outputs/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / clean
        target.write_bytes(content)
        return analyse_pcap(target, output=upload_dir / f"{target.stem}_analysis.json")

    @api.get("/alerts")
    def alerts(limit: int = 100) -> dict[str, Any]:
        limit = min(max(limit, 1), 1000)
        values = list(iter_jsonl("outputs/predictions.jsonl") or [])
        return {"count": min(len(values), limit), "alerts": values[-limit:]}

    @api.get("/incidents")
    def incidents() -> dict[str, Any]:
        graph = load_json("artifacts/reports/threat_graph.json", default={})
        return graph

    @api.get("/drift/status")
    def drift_status() -> dict[str, Any]:
        return get_engine().drift.status()

    @api.get("/audit/verify")
    def audit_verify() -> dict[str, Any]:
        return AuditChain().verify()

    @api.get("/models")
    def models() -> dict[str, Any]:
        registry = ModelRegistry()
        return {"active": load_json(registry.active_path, default={}), "models": registry.list()}

    @api.get("/explanations/{event_id}")
    def explanation(event_id: str) -> dict[str, Any]:
        path = resolve_path(Path("artifacts/explanations") / f"{safe_filename(event_id)}.json")
        value = load_json(path)
        if value is None:
            raise HTTPException(status_code=404, detail="Explanation not found")
        return value

    return api


app = create_app()
