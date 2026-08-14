from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


def _matrix_power(matrix: np.ndarray, power: float, eps: float = 1e-10) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, eps, None)
    return eigenvectors @ np.diag(eigenvalues**power) @ eigenvectors.T


def coral_fit(source: np.ndarray, target: np.ndarray, eps: float = 1e-3) -> dict[str, np.ndarray]:
    """Compute the CORAL (Sun & Saenko, 2016) whitening-recoloring transform that aligns
    the second-order statistics (and mean) of `source` features onto `target` features.

    Only unlabeled feature matrices are used (no target labels), so this is a valid
    unsupervised-domain-adaptation step: `target` may legitimately be validation/test
    feature data collected without ground-truth labels.
    """
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    dims = source.shape[1]
    cov_source = np.cov(source - source_mean, rowvar=False) + eps * np.eye(dims)
    cov_target = np.cov(target - target_mean, rowvar=False) + eps * np.eye(dims)
    whitening = _matrix_power(cov_source, -0.5)
    recoloring = _matrix_power(cov_target, 0.5)
    transform = whitening @ recoloring
    return {"transform": transform, "source_mean": source_mean, "target_mean": target_mean}


def coral_apply(features: np.ndarray, alignment: dict[str, np.ndarray]) -> np.ndarray:
    centered = features - alignment["source_mean"]
    return (centered @ alignment["transform"]) + alignment["target_mean"]


def recalibrate_batchnorm(model: Any, batches: Any, device: str) -> bool:
    """Adaptive Batch Normalization (Li et al., 2018): recompute BatchNorm running
    statistics using unlabeled target-domain forward passes, leaving all learned
    weights untouched. `batches` is an iterable of already-batched tensors (e.g. a
    DataLoader or a generator) rather than one concatenated tensor, so sequence
    data — which balloons by the window length — never needs to be materialized
    in memory all at once. Returns False if the model has no BatchNorm layers.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    import torch.nn as nn

    batchnorm_modules = [module for module in model.modules() if isinstance(module, nn.BatchNorm1d)]
    if not batchnorm_modules:
        return False
    for module in batchnorm_modules:
        module.reset_running_stats()
        module.momentum = None  # cumulative moving average over the full target pass
    was_training = model.training
    model.train()
    seen_any = False
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            if batch.shape[0] < 2:
                continue
            model(batch)
            seen_any = True
    for module in batchnorm_modules:
        module.momentum = 0.1
    model.train(was_training)
    model.eval()
    return seen_any


def enable_mc_dropout(model: Any) -> bool:
    """Force Dropout layers into train() mode while everything else stays in eval()
    mode, so repeated stochastic forward passes yield an MC-Dropout predictive
    distribution (Gal & Ghahramani, 2016) instead of a single point estimate.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    import torch.nn as nn

    found = False
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            found = True
    return found


def split_conformal_threshold(nonconformity_scores: np.ndarray, coverage: float = 0.9) -> float:
    """Split-conformal quantile (Vovk et al.): the (n+1)-adjusted empirical quantile
    of calibration nonconformity scores that guarantees marginal coverage on
    exchangeable future data, without any distributional assumption on the scores.
    """
    scores = np.sort(np.asarray(nonconformity_scores, dtype=float))
    n = len(scores)
    if n == 0:
        return 1.0
    rank = int(np.ceil((n + 1) * coverage))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])
