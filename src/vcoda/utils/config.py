from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from vcoda.utils.io import load_yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(*paths: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in paths:
        config = _deep_merge(config, load_yaml(path))
    return config


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
