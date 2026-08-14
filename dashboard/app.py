from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from vcoda.audit.chain import AuditChain
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.self_healing import SelfHealingManager
from vcoda.utils.io import iter_jsonl, load_json

st.set_page_config(page_title="V-CODA Security Operations", layout="wide")
st.title("V-CODA — Verifiable Cybersecurity-Oriented Detection and Adaptive Response")

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Alerts",
        "MITRE and Threat Graph",
        "Model Comparison",
        "Drift and Self-Healing",
        "Audit",
        "PCAP and Live Monitoring",
        "Settings",
    ],
)


def load_predictions() -> pd.DataFrame:
    rows = list(iter_jsonl("outputs/predictions.jsonl") or [])
    return pd.json_normalize(rows) if rows else pd.DataFrame()


AUTO_REFRESH = st.sidebar.toggle("Auto-refresh this page", value=True)
refresh_interval = f"{st.sidebar.slider('Refresh every (seconds)', 2, 30, 5)}s" if AUTO_REFRESH else None


def _fragment(run_every):
    """Apply st.fragment only when auto-refresh is enabled, so the toggle actually stops polling."""
    def decorator(func):
        return st.fragment(run_every=run_every)(func) if run_every else func
    return decorator


if page == "Overview":
    @_fragment(refresh_interval)
    def render_overview() -> None:
        predictions = load_predictions()
        if predictions.empty:
            st.info("No real predictions have been stored yet. Train the models and run `vcoda predict`, PCAP analysis, or live monitoring.")
        else:
            total = len(predictions)
            malicious = int((predictions.get("final_prediction", pd.Series(dtype=str)) == "attack").sum())
            critical = int((predictions.get("severity", pd.Series(dtype=str)) == "critical").sum())
            columns = st.columns(4)
            columns[0].metric("Processed flows", total)
            columns[1].metric("Attack predictions", malicious)
            columns[2].metric("Critical alerts", critical)
            columns[3].metric("Mean risk", round(float(predictions.get("risk_score", pd.Series([0])).mean()), 2))
            if "predicted_category" in predictions:
                st.subheader("Attacks by category")
                st.bar_chart(predictions["predicted_category"].value_counts())
            if "severity" in predictions:
                st.subheader("Severity distribution")
                st.bar_chart(predictions["severity"].value_counts())
            if "risk_score" in predictions:
                st.subheader("Risk timeline")
                st.line_chart(predictions[["risk_score"]])
            if "ensemble_status" in predictions:
                st.subheader("Ensemble usage")
                st.caption(
                    "Sequence-based deep models (cnn1d/cnn_lstm/transformer) need a 32-event warm-up "
                    "window per source before the full stacking ensemble can be used; new sources briefly "
                    "fall back to the standalone supervised model."
                )
                columns = st.columns(2)
                with columns[0]:
                    st.bar_chart(predictions["ensemble_status"].value_counts())
                with columns[1]:
                    stacking_share = float((predictions["ensemble_status"] != "supervised_fallback").mean())
                    st.metric("Full-ensemble coverage", f"{stacking_share:.0%}")

    render_overview()

elif page == "Alerts":
    @_fragment(refresh_interval)
    def render_alerts() -> None:
        predictions = load_predictions()
        if predictions.empty:
            st.info("No stored alerts.")
        else:
            category = st.sidebar.multiselect("Attack category", sorted(predictions.get("predicted_category", pd.Series(dtype=str)).dropna().unique()))
            severity = st.sidebar.multiselect("Severity", sorted(predictions.get("severity", pd.Series(dtype=str)).dropna().unique()))
            minimum_confidence = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.01)
            filtered = predictions.copy()
            if category:
                filtered = filtered[filtered["predicted_category"].isin(category)]
            if severity:
                filtered = filtered[filtered["severity"].isin(severity)]
            if "confidence" in filtered:
                filtered = filtered[filtered["confidence"] >= minimum_confidence]
            display_columns = [column for column in [
                "timestamp", "event_id", "source_ip", "destination_ip", "destination_port",
                "final_prediction", "predicted_category", "confidence", "anomaly_score", "risk_score",
                "severity", "ensemble_status", "model_disagreement", "uncertainty",
                "recommended_action", "response.executed", "drift_status.events",
            ] if column in filtered.columns]
            st.dataframe(filtered[display_columns], width="stretch")
            selected = st.selectbox("Inspect event", filtered.get("event_id", pd.Series(dtype=str)).tolist())
            if selected:
                row = filtered[filtered["event_id"] == selected].iloc[0].to_dict()
                st.json(row)

    render_alerts()

elif page == "MITRE and Threat Graph":
    @_fragment(refresh_interval)
    def render_mitre() -> None:
        graph = load_json("artifacts/reports/threat_graph.json", default={})
        if not graph:
            st.info("The threat graph is created from real detections. No graph data exists yet.")
        else:
            nodes = pd.DataFrame(graph.get("nodes", []))
            edges = pd.DataFrame(graph.get("edges", []))
            st.metric("Graph nodes", len(nodes))
            st.metric("Graph edges", len(edges))
            if not nodes.empty:
                st.subheader("Nodes")
                st.dataframe(nodes, width="stretch")
            if not edges.empty:
                st.subheader("Relationships")
                st.dataframe(edges, width="stretch")
        mapping = load_json("artifacts/reports/mitre_summary.json", default={})
        if mapping:
            st.json(mapping)

    render_mitre()

elif page == "Model Comparison":
    @_fragment(refresh_interval)
    def render_model_comparison() -> None:
        registry = ModelRegistry()
        active = load_json(registry.active_path, default={}).get("active", {})
        st.subheader("Active model leaderboard")
        st.caption("Every currently-promoted model, one row per task, read directly from the model registry.")
        if not active:
            st.info("No models are registered yet. Run `vcoda train ...` and `vcoda optimise-ensemble`.")
        else:
            leaderboard = []
            for task, record in active.items():
                metrics = record.get("metrics", {})
                leaderboard.append({
                    "task": task,
                    "model": record.get("model_name"),
                    "version": record.get("version"),
                    "macro_f1": metrics.get("macro_f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "false_positive_rate": metrics.get("false_positive_rate"),
                    "mcc": metrics.get("mcc"),
                })
            leaderboard_frame = pd.DataFrame(leaderboard).sort_values("macro_f1", ascending=False, na_position="last")
            st.dataframe(leaderboard_frame, width="stretch")
            if leaderboard_frame["macro_f1"].notna().any():
                st.subheader("Macro-F1 by task")
                st.bar_chart(leaderboard_frame.set_index("model")["macro_f1"])

        report = load_json("artifacts/reports/final_evaluation_report.json", default={})
        st.subheader("Held-out test-set evaluation (`vcoda evaluate`)")
        if not report.get("models"):
            st.info("Run `vcoda evaluate` after real model training. The dashboard does not fabricate comparison metrics.")
        else:
            table = []
            for name, metrics in report["models"].items():
                if "macro_f1" not in metrics:
                    continue
                table.append({
                    "model": name,
                    "macro_precision": metrics.get("macro_precision"),
                    "macro_recall": metrics.get("macro_recall"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "false_positive_rate": metrics.get("false_positive_rate"),
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "mcc": metrics.get("mcc"),
                })
            st.dataframe(pd.DataFrame(table), width="stretch")

        with st.expander("Raw active.json"):
            st.json(load_json(registry.active_path, default={}))

    render_model_comparison()

elif page == "Drift and Self-Healing":
    @_fragment(refresh_interval)
    def render_drift() -> None:
        status = load_json("artifacts/reports/self_healing_status.json", default={})
        if st.button("Run health and recovery check"):
            status = SelfHealingManager().run_once()
        st.subheader("Self-healing status")
        st.json(status or {"status": "not run"})
        events = list(iter_jsonl("artifacts/drift/drift_events.jsonl") or [])
        st.subheader("Drift events")
        if events:
            st.dataframe(pd.json_normalize(events), width="stretch")
        else:
            st.info("No drift events have been recorded.")

    render_drift()

elif page == "Audit":
    @_fragment(refresh_interval)
    def render_audit() -> None:
        verification = AuditChain().verify()
        st.json(verification)
        records = list(iter_jsonl("artifacts/audit/audit.jsonl") or [])
        if records:
            st.dataframe(pd.json_normalize(records[-200:]), width="stretch")
        else:
            st.info("The append-only audit chain is empty.")

    render_audit()

elif page == "PCAP and Live Monitoring":
    SEVERITY_BADGE = {
        "critical": "🔴 critical",
        "high": "🟠 high",
        "medium": "🟡 medium",
        "low": "🟢 low",
    }
    live_tab, pcap_tab = st.tabs(["Live monitoring", "PCAP analysis"])

    with live_tab:
        st.caption(
            "Streams `outputs/live_predictions.jsonl`, written by `vcoda monitor-live --interface <name>` "
            "(needs Npcap on Windows). Refreshes automatically every 3 seconds."
        )

        @st.fragment(run_every="3s")
        def render_live_feed() -> None:
            path = Path("outputs/live_predictions.jsonl")
            if not path.exists():
                st.info("No live capture running yet. Start one with `vcoda monitor-live --interface <name>`.")
                return
            rows = list(iter_jsonl(path) or [])
            if not rows:
                st.info("Capture is running, but no flows have completed yet.")
                return
            frame = pd.json_normalize(rows)
            last_seen = pd.to_datetime(frame["timestamp"]).max()
            age_seconds = (pd.Timestamp.now(tz="UTC") - last_seen).total_seconds()
            live = age_seconds < 30

            status_col, kpi = st.columns([1, 3])
            with status_col:
                if live:
                    st.success("🟢 Live", icon=None)
                else:
                    st.warning(f"⏸️ Idle ({age_seconds:.0f}s)", icon=None)
            with kpi:
                columns = st.columns(4)
                columns[0].metric("Flows captured", len(frame))
                columns[1].metric("Attacks flagged", int((frame.get("final_prediction") == "attack").sum()))
                columns[2].metric("Critical alerts", int((frame.get("severity") == "critical").sum()))
                if "ensemble_status" in frame:
                    coverage = float((frame["ensemble_status"] != "supervised_fallback").mean())
                    columns[3].metric("Full-ensemble coverage", f"{coverage:.0%}")

            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.caption("Top talkers (source IP)")
                st.bar_chart(frame["source_ip"].value_counts().head(8))
            with chart_columns[1]:
                st.caption("Severity mix")
                st.bar_chart(frame["severity"].value_counts())

            st.caption("Most recent flows (newest first)")
            recent = frame.sort_values("timestamp", ascending=False).head(25).copy()
            recent["severity"] = recent["severity"].map(lambda s: SEVERITY_BADGE.get(s, s))
            display_columns = [column for column in [
                "timestamp", "source_ip", "destination_ip", "destination_port",
                "final_prediction", "predicted_category", "confidence", "risk_score",
                "severity", "ensemble_status",
            ] if column in recent.columns]
            st.dataframe(recent[display_columns], width="stretch", hide_index=True)

        render_live_feed()

    with pcap_tab:
        st.caption("Analyse a saved capture with `vcoda analyse-pcap --input <file.pcap>`.")
        reports = sorted(Path("outputs").glob("*pcap*analysis*.json"))
        if reports:
            selected = st.selectbox("PCAP report", reports)
            report = load_json(selected, default={})
            results = report.get("results", [])
            if results:
                columns = st.columns(3)
                columns[0].metric("Flows analysed", report.get("flow_count", len(results)))
                columns[1].metric("Feature coverage", f"{report.get('feature_compatibility', {}).get('coverage', 0):.0%}")
                statuses = pd.Series([r.get("ensemble_status") for r in results]).value_counts()
                columns[2].metric("Full-ensemble predictions", int(statuses.get("stacking", 0)))
                st.bar_chart(statuses)
            st.json(report)
        else:
            st.info("No PCAP analysis reports found in `outputs/` yet.")
        feature_reports = sorted(Path("outputs").glob("*compatibility*.json"))
        if feature_reports:
            st.subheader("Feature compatibility")
            st.json(load_json(feature_reports[-1], default={}))

elif page == "Settings":
    st.warning("Dashboard settings are read-only. Edit version-controlled YAML files and restart services.")
    for config in ["configs/default.yaml", "configs/inference.yaml", "configs/response_policy.yaml", "configs/self_healing.yaml"]:
        st.subheader(config)
        st.code(Path(config).read_text(encoding="utf-8"), language="yaml")
