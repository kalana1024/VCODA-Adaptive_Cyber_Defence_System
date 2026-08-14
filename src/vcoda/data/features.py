from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float).abs().clip(lower=epsilon)


def derive_flow_features(frame: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    result = frame.copy()
    required = {"in_bytes", "out_bytes", "in_packets", "out_packets", "flow_duration_ms"}
    if not required.issubset(result.columns):
        return result
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["total_bytes"] = result["in_bytes"].fillna(0) + result["out_bytes"].fillna(0)
    result["total_packets"] = result["in_packets"].fillna(0) + result["out_packets"].fillna(0)
    result["bytes_per_packet"] = _safe_divide(result["total_bytes"], result["total_packets"], epsilon)
    duration_seconds = result["flow_duration_ms"].fillna(0) / 1000.0
    result["packets_per_second"] = _safe_divide(result["total_packets"], duration_seconds, epsilon)
    result["bytes_per_second"] = _safe_divide(result["total_bytes"], duration_seconds, epsilon)
    result["direction_byte_ratio"] = _safe_divide(result["in_bytes"].fillna(0), result["out_bytes"].fillna(0), epsilon)
    result["direction_packet_ratio"] = _safe_divide(result["in_packets"].fillna(0), result["out_packets"].fillna(0), epsilon)
    result["packet_asymmetry"] = (result["in_packets"].fillna(0) - result["out_packets"].fillna(0)).abs()
    result["byte_asymmetry"] = (result["in_bytes"].fillna(0) - result["out_bytes"].fillna(0)).abs()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
