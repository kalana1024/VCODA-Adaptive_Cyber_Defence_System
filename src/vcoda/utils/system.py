from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import psutil

from vcoda.utils.io import dump_json, resolve_path

OPTIONAL_MODULES = [
    "xgboost", "lightgbm", "catboost", "optuna", "shap", "torch", "river",
    "fastapi", "streamlit", "scapy", "pyshark", "mlflow", "pyarrow",
]


def detect_hardware() -> dict[str, Any]:
    gpu: dict[str, Any] = {"cuda_available": False, "name": None, "count": 0}
    try:
        import torch
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["count"] = int(torch.cuda.device_count())
        if gpu["cuda_available"]:
            gpu["name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        gpu["detection_error"] = type(exc).__name__
    disk = psutil.disk_usage(str(Path.cwd().anchor or Path.cwd()))
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "gpu": gpu,
        "docker": shutil.which("docker") is not None,
        "tshark": shutil.which("tshark") is not None,
        "git": shutil.which("git") is not None,
        "modules": {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_MODULES},
        "is_admin": _is_admin(),
    }


def choose_profile(hardware: dict[str, Any]) -> str:
    ram = float(hardware["ram_gb"])
    gpu = bool(hardware["gpu"]["cuda_available"])
    if gpu and ram >= 32:
        return "high"
    if ram >= 16:
        return "medium"
    return "small"


def system_check(output: str | Path = "artifacts/reports/system_check.json") -> dict[str, Any]:
    report = detect_hardware()
    report["recommended_profile"] = choose_profile(report)
    report["project_root"] = str(resolve_path("."))
    dump_json(report, output)
    return report


def _is_admin() -> bool:
    if os.name != "nt":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False
