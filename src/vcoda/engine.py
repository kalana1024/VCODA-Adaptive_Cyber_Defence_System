from __future__ import annotations

import json
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vcoda.anomaly.models import AnomalyBundle, OnlineAnomalyDetector
from vcoda.audit.chain import AuditChain
from vcoda.drift.monitor import DriftMonitor
from vcoda.ensemble.fusion import EnsembleBundle
from vcoda.explainability.engine import ExplainabilityEngine
from vcoda.mitre.reasoner import MitreReasoner, ThreatGraph
from vcoda.models.bundle import ModelBundle
from vcoda.models.deep import DeepBundle, MultiTaskBundle
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.heartbeat import write_heartbeat
from vcoda.monitoring.self_healing import SelfHealingManager
from vcoda.response.engine import ResponseEngine
from vcoda.risk import risk_score, severity_band
from vcoda.threat_intel.engine import ThreatIntelEngine
from vcoda.utils.io import append_jsonl, load_yaml, resolve_path

SEQUENCE_ARCHITECTURES = {"cnn1d", "cnn_lstm", "transformer"}


class VCODAEngine:
    def __init__(
        self,
        inference_config: str | Path = "configs/inference.yaml",
        response_policy: str | Path = "configs/response_policy.yaml",
        self_healing: bool = True,
    ) -> None:
        self.config = load_yaml(inference_config)
        self.registry = ModelRegistry()
        self.audit = AuditChain()
        self.response = ResponseEngine(response_policy)
        self.mitre = MitreReasoner()
        self.graph = ThreatGraph()
        cti_config = self.config.get("threat_intel", {})
        self.cti = ThreatIntelEngine(
            local_ioc_files=cti_config.get("local_ioc_files", []),
            cache_seconds=int(cti_config.get("cache_seconds", 3600)),
            request_timeout_seconds=float(cti_config.get("request_timeout_seconds", 10)),
        )
        self.use_external_cti = bool(cti_config.get("use_external_services", False))
        self.drift = DriftMonitor()
        self.explain = ExplainabilityEngine()
        self.healer = SelfHealingManager() if self_healing else None
        self.supervised_binary: ModelBundle | None = None
        self.supervised_multiclass: ModelBundle | None = None
        self.deep_binary: DeepBundle | None = None
        self.anomaly: AnomalyBundle | None = None
        self.ensemble: EnsembleBundle | None = None
        self.online_anomaly: OnlineAnomalyDetector | None = None
        self.deep_members: dict[str, DeepBundle | MultiTaskBundle] = {}
        self._sequence_buffers: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=32))
        self._load_active_models()

    def _load_active_models(self) -> None:
        loaders = {
            "supervised_binary": ("supervised_binary", ModelBundle.load, "supervised_binary"),
            "supervised_multiclass": ("supervised_multiclass", ModelBundle.load, "supervised_multiclass"),
            "deep_binary": ("deep_binary", DeepBundle.load, "deep_binary"),
            "anomaly": ("anomaly_offline", AnomalyBundle.load, "anomaly"),
            "ensemble": ("ensemble_binary", EnsembleBundle.load, "ensemble"),
        }
        for _, (task, loader, attribute) in loaders.items():
            record = self.registry.active(task)
            if not record:
                continue
            try:
                loaded = loader(resolve_path(record["artifact_path"]), record["artifact_sha256"])
            except Exception:
                if self.healer:
                    self.healer.run_once()
                    record = self.registry.active(task)
                    loaded = loader(resolve_path(record["artifact_path"]), record["artifact_sha256"]) if record else None
                else:
                    raise
            setattr(self, attribute, loaded)
        feature_names: list[str] = []
        if self.supervised_binary and self.supervised_binary.preprocessor.schema:
            feature_names = self.supervised_binary.preprocessor.schema.numeric[:20]
        self.online_anomaly = OnlineAnomalyDetector(feature_names)
        self._load_ensemble_members()

    def _load_ensemble_members(self) -> None:
        """Individually load every deep-learning bundle the active ensemble needs.
        Only one architecture can hold the shared "deep_binary" active slot at a
        time, but the ensemble may reference several (mlp, cnn1d, cnn_lstm,
        transformer, autoencoder, multitask) simultaneously — so each is fetched by
        its own model_name via ModelRegistry.latest() instead of the active-task
        lookup used for single-model slots."""
        self.deep_members = {}
        if self.ensemble is None:
            return
        for member_name in self.ensemble.model_names:
            registry_name = member_name.removeprefix("deep_")
            record = self.registry.latest(registry_name)
            if not record:
                continue
            architecture = str(record.get("hyperparameters", {}).get("architecture", ""))
            try:
                if architecture == "multitask":
                    bundle: DeepBundle | MultiTaskBundle = MultiTaskBundle.load(
                        resolve_path(record["artifact_path"]), record["artifact_sha256"]
                    )
                else:
                    bundle = DeepBundle.load(resolve_path(record["artifact_path"]), record["artifact_sha256"])
            except Exception:
                continue
            self.deep_members[member_name] = bundle

    def status(self) -> dict[str, Any]:
        return {
            "supervised_binary": self.registry.active("supervised_binary"),
            "supervised_multiclass": self.registry.active("supervised_multiclass"),
            "deep_binary": self.registry.active("deep_binary"),
            "anomaly_offline": self.registry.active("anomaly_offline"),
            "ensemble_binary": self.registry.active("ensemble_binary"),
            "drift": self.drift.status(),
            "response_mode": self.response.policy.get("mode"),
        }

    def predict(self, event: dict[str, Any], *, explain: bool = True, analyst_confirmed: bool = False) -> dict[str, Any]:
        if self.supervised_binary is None:
            raise RuntimeError("No active supervised binary model. Train and promote one before inference.")
        event_id = str(event.get("event_id") or uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        frame = pd.DataFrame([event])
        schema = self.supervised_binary.schema_report(frame)
        min_coverage = float(self.config["inference"].get("minimum_feature_coverage", 0.70))
        if schema["coverage"] < min_coverage and self.config["inference"].get("reject_on_schema_mismatch", False):
            raise ValueError(f"Feature coverage {schema['coverage']:.2%} is below the configured minimum")

        model_probabilities: dict[str, float] = {}
        supervised_probability = float(self.supervised_binary.predict_proba(frame)[0, 1])
        active_sup = self.registry.active("supervised_binary") or {}
        sup_aliases = {"supervised_binary", str(active_sup.get("model_name", "")).replace("_", "_")}
        for alias in sup_aliases:
            if alias:
                model_probabilities[alias] = supervised_probability
        category = "attack" if supervised_probability >= (self.supervised_binary.threshold or 0.5) else "benign"
        category_probability = supervised_probability
        if self.supervised_multiclass is not None:
            multiclass = self.supervised_multiclass.predict_proba(frame)[0]
            category_index = int(np.argmax(multiclass))
            category = self.supervised_multiclass.classes[category_index]
            category_probability = float(multiclass[category_index])

        if self.deep_binary is not None and self.deep_binary.architecture == "mlp":
            deep_probability = float(self.deep_binary.predict_proba(frame)[0, 1])
            active_deep = self.registry.active("deep_binary") or {}
            for alias in {"deep_binary", f"deep_{active_deep.get('model_name', '')}", str(active_deep.get("model_name", ""))}:
                if alias:
                    model_probabilities[alias] = deep_probability
        model_probabilities.update(self._predict_deep_members(event, frame))

        anomaly_confidence = 0.0
        anomaly_explanation: list[dict[str, float]] = []
        if self.anomaly is not None:
            anomaly_confidence = float(self.anomaly.confidence(frame)[0])
            anomaly_explanation = self.anomaly.explain(frame)[0]
            active_anomaly = self.registry.active("anomaly_offline") or {}
            for alias in {"anomaly_offline", f"anomaly_{active_anomaly.get('model_name', '')}", str(active_anomaly.get("model_name", ""))}:
                if alias:
                    model_probabilities[alias] = anomaly_confidence
        online_score = self.online_anomaly.score(event) if self.online_anomaly else 0.0
        anomaly_confidence = max(anomaly_confidence, online_score)

        disagreement = float(np.std(list(model_probabilities.values()))) if model_probabilities else 0.0
        uncertainty = min(disagreement / 0.25, 1.0)
        final_probability = supervised_probability
        ensemble_status = "supervised_fallback"
        if self.ensemble is not None and all(name in model_probabilities for name in self.ensemble.model_names):
            combined = self.ensemble.combine({name: np.array([model_probabilities[name]]) for name in self.ensemble.model_names})
            final_probability = float(combined["probability"][0])
            disagreement = float(combined["disagreement"][0])
            uncertainty = float(combined["uncertainty"][0])
            ensemble_status = self.ensemble.method

        cti_result = self.cti.enrich(
            str(event.get("cti_text", "")),
            event.get("indicators"),
            use_external=self.use_external_cti,
        )
        evidence = self._evidence(event)
        mitre = self.mitre.map(category, evidence) if category.lower() != "benign" else []
        model_agreement = max(0.0, 1.0 - disagreement * 2)
        risk = risk_score(
            confidence=final_probability, anomaly=anomaly_confidence, attack_category=category,
            asset_criticality=str(event.get("asset_criticality", "medium")),
            repeated_event_count=int(event.get("repeated_event_count", 0)),
            ioc_reputation=float(cti_result.get("reputation", 0.0)), model_agreement=model_agreement,
            uncertainty=uncertainty, temporal_correlation=float(event.get("temporal_correlation", 0.0)),
            weights=self.config["risk"]["weights"],
        )
        severity = severity_band(risk)
        corroborating_signals = sum([supervised_probability >= (self.supervised_binary.threshold or 0.5), anomaly_confidence >= 0.8, bool(cti_result.get("matches")), bool(mitre)])
        action = self.response.recommend(severity, category, event)
        decision = self.response.decide(
            action=action, source_ip=event.get("source_ip") or event.get("src_ip"),
            confidence=final_probability, risk_score=risk, corroborating_signals=corroborating_signals,
            asset_type=event.get("asset_type"), analyst_confirmed=analyst_confirmed,
        )
        response_result = self.response.execute(decision, event.get("source_ip") or event.get("src_ip"))
        drift_events = self.drift.update(
            event_id=event_id, confidence=final_probability, anomaly_score=anomaly_confidence,
            predicted_attack=final_probability >= (self.supervised_binary.threshold or 0.5),
            error=event.get("verified_error"), features={k: float(v) for k, v in event.items() if isinstance(v, (int, float))},
        )
        if self.online_anomaly:
            self.online_anomaly.learn(event, safe=risk < 20 and not drift_events)

        explanation = {
            "supervised": self.explain.explain_supervised(self.supervised_binary, frame) if explain else [],
            "anomaly": anomaly_explanation if explain else [],
            "reason_codes": [name for name, active in evidence.items() if active],
            "model_disagreement": disagreement,
            "uncertainty": uncertainty,
            "feature_coverage": schema,
            "mc_dropout_uncertainty": self._mc_dropout_uncertainty(frame) if explain else None,
        }
        result = {
            "event_id": event_id, "timestamp": timestamp,
            "source_ip": event.get("source_ip") or event.get("src_ip"),
            "destination_ip": event.get("destination_ip") or event.get("dst_ip"),
            "destination_port": event.get("destination_port") or event.get("dst_port"),
            "asset_id": event.get("asset_id"), "asset_type": event.get("asset_type"),
            "final_prediction": "attack" if final_probability >= (self.ensemble.threshold if self.ensemble else (self.supervised_binary.threshold or 0.5)) else "benign",
            "predicted_category": category, "category_confidence": category_probability,
            "confidence": final_probability, "severity": severity, "risk_score": risk,
            "anomaly_score": anomaly_confidence, "model_probabilities": model_probabilities,
            "ensemble_status": ensemble_status, "model_disagreement": disagreement, "uncertainty": uncertainty,
            "mitre": mitre, "threat_intel": cti_result, "explanation": explanation,
            "drift_events": drift_events, "drift_status": self.drift.status(),
            "recommended_action": action, "response": response_result.__dict__,
            "model_versions": self.status(),
        }
        incident_id = self.graph.add_detection(result)
        result["incident_id"] = incident_id
        result["correlated_events"] = self.graph.correlated_events(event_id)
        audit_record = self.audit.append(result)
        result["audit_sequence"] = audit_record["sequence"]
        result["audit_hash"] = audit_record["record_hash"]
        if explain:
            result["explanation_path"] = str(self.explain.save(event_id, explanation))
        append_jsonl(self.config["inference"].get("prediction_store", "outputs/predictions.jsonl"), result)
        self.graph.export()
        write_heartbeat("inference", {"last_event_id": event_id})
        return result

    def _mc_dropout_uncertainty(self, frame: pd.DataFrame, samples: int = 20) -> dict[str, float] | None:
        """Epistemic uncertainty via MC-Dropout on the mlp ensemble member (the
        cheapest non-sequence deep model with active Dropout layers): repeated
        stochastic forward passes give a predictive std alongside the mean,
        rather than a single point estimate. Skipped on the live/pcap batch path
        (explain=False there) since it costs `samples` extra forward passes."""
        bundle = self.deep_members.get("deep_mlp_binary")
        if bundle is None or not isinstance(bundle, DeepBundle):
            return None
        try:
            result = bundle.predict_proba_mc(frame, samples=samples)
        except Exception:
            return None
        return {"mean_attack_probability": float(result["mean"][0, 1]), "std_attack_probability": float(result["std"][0, 1])}

    def _predict_deep_members(self, event: dict[str, Any], frame: pd.DataFrame) -> dict[str, float]:
        """Score every loaded ensemble member. Non-sequence architectures (mlp,
        autoencoder, multitask) score the single incoming event directly. Sequence
        architectures (cnn1d, cnn_lstm, transformer) need a 32-event window, so a
        rolling per-source buffer is maintained here; a source only gets scored by
        those members once enough history has accumulated for it — a fresh source
        legitimately falls back to the non-sequence members until then."""
        probabilities: dict[str, float] = {}
        if not self.deep_members:
            return probabilities
        source_key = str(event.get("source_ip") or event.get("src_ip") or "unknown")
        buffer = self._sequence_buffers[source_key]
        buffer.append(dict(event))
        for member_name, bundle in self.deep_members.items():
            try:
                if isinstance(bundle, MultiTaskBundle):
                    probabilities[member_name] = float(bundle.predict_proba(frame)["binary"][0, 1])
                elif bundle.architecture in SEQUENCE_ARCHITECTURES:
                    if len(buffer) < bundle.sequence_length:
                        continue
                    window = pd.DataFrame(list(buffer)[-bundle.sequence_length :])
                    transformed = bundle.preprocessor.transform(window)
                    sequence_array = transformed.reshape(1, *transformed.shape)
                    probabilities[member_name] = float(bundle.predict_sequences(sequence_array)[0, 1])
                else:
                    probabilities[member_name] = float(bundle.predict_proba(frame)[0, 1])
            except Exception:
                continue
        return probabilities

    @staticmethod
    def _evidence(event: dict[str, Any]) -> dict[str, bool]:
        packet_rate = float(event.get("packets_per_second", event.get("packet_rate", 0)) or 0)
        destination_ports = int(event.get("unique_destination_ports", 1) or 1)
        destination_hosts = int(event.get("unique_destination_ips", 1) or 1)
        failures = int(event.get("failed_connections", 0) or 0)
        return {
            "high_packet_rate": packet_rate > 5000,
            "high_connection_rate": float(event.get("connections_per_second", 0) or 0) > 100,
            "many_destination_ports": destination_ports >= 20,
            "many_destination_hosts": destination_hosts >= 10,
            "short_failed_connections": failures >= 10,
            "repeated_authentication_failures": failures >= 10,
            "web_exploit_classification": str(event.get("service", "")).lower() in {"http", "https"} and bool(event.get("web_attack_indicator")),
            "suspicious_http_pattern": bool(event.get("suspicious_http_pattern")),
        }
