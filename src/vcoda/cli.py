from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich import print as rprint

from vcoda.anomaly.models import train_anomaly_models
from vcoda.audit.chain import AuditChain
from vcoda.capture.live import monitor_live
from vcoda.capture.pcap import analyse_pcap
from vcoda.data.inspect import inspect_dataset
from vcoda.data.prepare import prepare_dataset
from vcoda.engine import VCODAEngine
from vcoda.ensemble.fusion import optimise_ensemble
from vcoda.models.deep import search_deep_hyperparameters, train_deep, train_deep_multitask
from vcoda.models.registry import ModelRegistry
from vcoda.models.supervised import train_supervised
from vcoda.monitoring.self_healing import SelfHealingManager
from vcoda.reports.evaluation import evaluate_active_models
from vcoda.utils.io import dump_json, load_json, resolve_path
from vcoda.utils.system import system_check

app = typer.Typer(help="V-CODA cybersecurity AI command line")
train_app = typer.Typer(help="Train versioned V-CODA model layers")
app.add_typer(train_app, name="train")


def _show(value: Any) -> None:
    rprint(json.dumps(value, indent=2, default=str))


@app.command("system-check")
def cli_system_check(output: str = "artifacts/reports/system_check.json") -> None:
    """Inspect Windows/Linux hardware, dependencies, GPU, Docker and capture tools."""
    _show(system_check(output))


@app.command("inspect-data")
def cli_inspect_data(
    data_dir: str = typer.Option("data/raw/nf_uq_nids_v2", help="Directory containing real NF-UQ-NIDS-v2 files"),
    config: str = "configs/training.yaml",
    full_scan: bool = False,
) -> None:
    """Inspect actual dataset files and generate a schema report before transformation."""
    from vcoda.utils.io import load_yaml, dump_yaml
    cfg = load_yaml(config)
    cfg["dataset"]["data_dir"] = data_dir
    temporary = "artifacts/reports/runtime_training_config.yaml"
    dump_yaml(cfg, temporary)
    _show(inspect_dataset(temporary, full_scan=full_scan))


@app.command("prepare-data")
def cli_prepare_data(
    config: str = "configs/training.yaml",
    features: str = "configs/features.yaml",
) -> None:
    """Clean, leakage-check, split and cache the inspected real dataset in chunks."""
    _show(prepare_dataset(config, features))


@train_app.command("supervised")
def cli_train_supervised(
    task: str = typer.Option("binary", help="binary or multiclass"),
    config: str = "configs/training.yaml",
    models: str = typer.Option("", help="Comma-separated model names; empty uses configuration"),
    optuna_trials: int | None = None,
) -> None:
    names = [value.strip() for value in models.split(",") if value.strip()] or None
    _show(train_supervised(config, task=task, model_names=names, optuna_trials=optuna_trials))


@train_app.command("deep")
def cli_train_deep(
    architecture: str = typer.Option("mlp", help="mlp, cnn1d, cnn_lstm, transformer or autoencoder"),
    task: str = typer.Option("binary", help="binary or multiclass"),
    config: str = "configs/training.yaml",
    optuna_trials: int = typer.Option(0, help="Quick Optuna search over lr/weight_decay/gradient_clip before the full run"),
) -> None:
    overrides = search_deep_hyperparameters(architecture, config, task=task, trials=optuna_trials) if optuna_trials > 0 else None
    _show(train_deep(architecture, config, task=task, hyperparameter_overrides=overrides))


@train_app.command("multitask")
def cli_train_multitask(config: str = "configs/training.yaml") -> None:
    """Train the shared-trunk binary+multiclass multi-task deep model."""
    _show(train_deep_multitask(config))


@train_app.command("anomaly")
def cli_train_anomaly(
    config: str = "configs/training.yaml",
    models: str = typer.Option("", help="Comma-separated: isolation_forest, local_outlier_factor, one_class_svm"),
) -> None:
    names = [value.strip() for value in models.split(",") if value.strip()] or None
    _show(train_anomaly_models(config, names))


@app.command("optimise-ensemble")
def cli_optimise_ensemble(config: str = "configs/training.yaml") -> None:
    """Fit stacking or validation-optimised soft-voting fusion from aligned model outputs."""
    _show(optimise_ensemble(config))


@app.command("evaluate")
def cli_evaluate(config: str = "configs/training.yaml") -> None:
    """Evaluate active models using the untouched prepared test partition."""
    _show(evaluate_active_models(config))


@app.command("predict")
def cli_predict(
    input: str = typer.Option(..., "--input", help="CSV, JSON or JSONL flow file"),
    output: str = "outputs/batch_prediction.json",
    explain: bool = True,
) -> None:
    path = resolve_path(input)
    if not path.exists():
        raise typer.BadParameter(f"Input does not exist: {path}")
    if path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = load_json(path)
        rows = raw if isinstance(raw, list) else [raw]
    engine = VCODAEngine()
    results = [engine.predict(row, explain=explain) for row in rows]
    report = {"input": str(path), "count": len(results), "results": results}
    dump_json(report, output)
    _show({"output": output, "count": len(results)})


@app.command("analyse-pcap")
def cli_analyse_pcap(
    input: str = typer.Option(..., "--input", help="PCAP or PCAPNG file"),
    output: str = "outputs/pcap_analysis.json",
    maximum_packets: int | None = None,
) -> None:
    _show(analyse_pcap(input, output, maximum_packets))


@app.command("monitor-live")
def cli_monitor_live(
    interface: str = typer.Option(..., "--interface", help="Npcap/Scapy interface name or index"),
    flush_seconds: int = 10,
) -> None:
    monitor_live(interface, flush_seconds=flush_seconds)


@app.command("serve-api")
def cli_serve_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI backend."""
    import uvicorn
    uvicorn.run("vcoda.api.app:app", host=host, port=port, reload=reload)


@app.command("dashboard")
def cli_dashboard(port: int = 8501) -> None:
    """Start the Streamlit dashboard using real stored results."""
    command = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(port)]
    raise typer.Exit(subprocess.run(command, check=False).returncode)


@app.command("list-models")
def cli_list_models(task: str | None = None) -> None:
    registry = ModelRegistry()
    _show({"active": load_json(registry.active_path, default={}), "models": registry.list(task)})


@app.command("promote-model")
def cli_promote_model(model_name: str, version: str) -> None:
    _show(ModelRegistry().promote(model_name, version))


@app.command("rollback-model")
def cli_rollback_model(task: str) -> None:
    _show(ModelRegistry().rollback(task))


@app.command("verify-audit")
def cli_verify_audit() -> None:
    _show(AuditChain().verify())


@app.command("heal")
def cli_heal(watch: bool = False) -> None:
    manager = SelfHealingManager()
    if watch:
        manager.watch()
    else:
        _show(manager.run_once())


if __name__ == "__main__":
    app()
