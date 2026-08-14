from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from vcoda.data.loaders import load_manifest, load_split, split_xy
from vcoda.models.domain_adaptation import (
    coral_apply,
    coral_fit,
    enable_mc_dropout,
    recalibrate_batchnorm,
)
from vcoda.models.metrics import choose_binary_threshold, evaluate_classification
from vcoda.models.preprocessing import VCODAPreprocessor
from vcoda.models.registry import ModelRegistry
from vcoda.monitoring.experiments import ExperimentTracker
from vcoda.utils.io import dump_json, load_yaml, resolve_path, sha256_file


def _torch() -> Any:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is not installed. Install requirements/gpu.txt or the CPU PyTorch package.")
    import torch
    return torch


class FocalLoss:
    def __init__(self, alpha: Any = None, gamma: float = 2.0) -> None:
        torch = _torch()
        self.torch = torch
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits: Any, targets: Any) -> Any:
        torch = self.torch
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_network(architecture: str, input_dim: int, classes: int, profile: str = "medium", sequence_length: int = 32) -> Any:
    torch = _torch()
    nn = torch.nn
    sizes = {
        "small": {"hidden": 64, "channels": 32, "layers": 1, "heads": 2},
        "medium": {"hidden": 128, "channels": 64, "layers": 2, "heads": 4},
        "high": {"hidden": 256, "channels": 128, "layers": 3, "heads": 8},
    }[profile]

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = sizes["hidden"]
            self.network = nn.Sequential(
                nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.30),
                nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.GELU(), nn.Dropout(0.20),
                nn.Linear(h // 2, classes),
            )
        def forward(self, x: Any) -> Any:
            return self.network(x)

    class CNN1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = sizes["channels"]
            self.features = nn.Sequential(
                nn.Conv1d(input_dim, c, 3, padding=1), nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c, c * 2, 3, padding=1), nn.BatchNorm1d(c * 2), nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(c * 2, classes))
        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x.transpose(1, 2)))

    class CNNLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = sizes["channels"]
            h = sizes["hidden"]
            self.conv = nn.Sequential(nn.Conv1d(input_dim, c, 3, padding=1), nn.GELU(), nn.BatchNorm1d(c))
            self.lstm = nn.LSTM(c, h, num_layers=sizes["layers"], batch_first=True, bidirectional=True, dropout=0.2 if sizes["layers"] > 1 else 0.0)
            self.out = nn.Sequential(nn.Dropout(0.25), nn.Linear(h * 2, classes))
        def forward(self, x: Any) -> Any:
            encoded = self.conv(x.transpose(1, 2)).transpose(1, 2)
            sequence, _ = self.lstm(encoded)
            return self.out(sequence[:, -1, :])

    class TransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = sizes["hidden"]
            self.embedding = nn.Linear(input_dim, h)
            self.position = nn.Parameter(torch.zeros(1, sequence_length, h))
            encoder_layer = nn.TransformerEncoderLayer(d_model=h, nhead=sizes["heads"], dim_feedforward=h * 2, dropout=0.2, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=sizes["layers"])
            self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, classes))
        def forward(self, x: Any) -> Any:
            embedded = self.embedding(x) + self.position[:, : x.shape[1], :]
            return self.out(self.encoder(embedded).mean(dim=1))

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = sizes["hidden"]
            latent = max(8, h // 8)
            self.encoder = nn.Sequential(nn.Linear(input_dim, h), nn.GELU(), nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, latent))
            self.decoder = nn.Sequential(nn.Linear(latent, h // 2), nn.GELU(), nn.Linear(h // 2, h), nn.GELU(), nn.Linear(h, input_dim))
        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    mapping = {"mlp": MLP, "cnn1d": CNN1D, "cnn_lstm": CNNLSTM, "transformer": TransformerModel, "autoencoder": Autoencoder}
    if architecture not in mapping:
        raise ValueError(f"Unknown deep architecture: {architecture}")
    return mapping[architecture]()


def build_multitask_network(input_dim: int, binary_classes: int, multiclass_classes: int, profile: str = "medium") -> Any:
    """Shared-trunk multi-task network: one encoder feeding a binary head and a
    multiclass head simultaneously, so both objectives regularise the same
    representation instead of training two unrelated models.
    """
    torch = _torch()
    nn = torch.nn
    sizes = {
        "small": {"hidden": 64},
        "medium": {"hidden": 128},
        "high": {"hidden": 256},
    }[profile]

    class MultiTaskMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = sizes["hidden"]
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.30),
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.20),
            )
            self.binary_head = nn.Linear(h, binary_classes)
            self.multiclass_head = nn.Linear(h, multiclass_classes)

        def forward(self, x: Any) -> tuple[Any, Any]:
            features = self.trunk(x)
            return self.binary_head(features), self.multiclass_head(features)

    return MultiTaskMLP()


def _make_sequences(x: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, length: int, require_order: bool) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if "sequence_group" not in metadata or "event_time" not in metadata:
        if require_order:
            raise RuntimeError("Sequence models require sequence_group and event_time metadata from dataset preparation")
        metadata = metadata.copy()
        metadata["sequence_group"] = "all"
        metadata["event_time"] = np.arange(len(metadata))
    order = metadata.assign(_position=np.arange(len(metadata))).sort_values(["sequence_group", "event_time", "_position"])
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    rows: list[pd.Series] = []
    for _, group in order.groupby("sequence_group", sort=False):
        positions = group["_position"].to_numpy()
        if len(positions) < length:
            continue
        for end in range(length - 1, len(positions)):
            selected = positions[end - length + 1 : end + 1]
            sequences.append(x[selected])
            targets.append(int(y[selected[-1]]))
            rows.append(metadata.iloc[selected[-1]])
    if not sequences:
        raise RuntimeError("No valid temporal sequences were produced. Check ordering/group configuration.")
    return np.stack(sequences).astype(np.float32), np.asarray(targets, dtype=np.int64), pd.DataFrame(rows).reset_index(drop=True)


class TemporalSequenceDataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metadata: pd.DataFrame,
        length: int,
        require_order: bool,
    ) -> None:
        self.x = x
        self.y = y
        self.length = length
        self.require_order = require_order
        metadata = metadata.copy()
        if "sequence_group" not in metadata or "event_time" not in metadata:
            if require_order:
                raise RuntimeError("Sequence models require sequence_group and event_time metadata from dataset preparation")
            metadata["sequence_group"] = "all"
            metadata["event_time"] = np.arange(len(metadata))

        ordered = metadata.assign(_position=np.arange(len(metadata))).sort_values(
            ["sequence_group", "event_time", "_position"], kind="stable"
        )
        self.positions = ordered["_position"].to_numpy(dtype=np.int64)
        self._sequence_start_offsets: list[int] = []
        self._sequence_target_positions: list[int] = []
        rows: list[pd.Series] = []
        offset = 0
        for _, group in ordered.groupby("sequence_group", sort=False):
            group_positions = group["_position"].to_numpy(dtype=np.int64)
            if len(group_positions) < length:
                offset += len(group_positions)
                continue
            for end_index in range(length - 1, len(group_positions)):
                self._sequence_start_offsets.append(offset + end_index - length + 1)
                self._sequence_target_positions.append(int(group_positions[end_index]))
                rows.append(metadata.iloc[int(group_positions[end_index])])
            offset += len(group_positions)

        if not self._sequence_start_offsets:
            raise RuntimeError("No valid temporal sequences were produced. Check ordering/group configuration.")

        self.sequence_start_offsets = np.asarray(self._sequence_start_offsets, dtype=np.int64)
        self.sequence_target_positions = np.asarray(self._sequence_target_positions, dtype=np.int64)
        self.targets = np.asarray(self.y[self.sequence_target_positions], dtype=np.int64)
        self.sequence_metadata = pd.DataFrame(rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.sequence_start_offsets)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = _torch()
        start = int(self.sequence_start_offsets[index])
        indices = self.positions[start : start + self.length]
        sequence = self.x[indices]
        target = int(self.targets[index])
        return torch.as_tensor(sequence, dtype=torch.float32), torch.as_tensor(target, dtype=torch.long)


@dataclass
class DeepBundle:
    architecture: str
    task: str
    classes: list[str]
    input_dim: int
    profile: str
    sequence_length: int
    preprocessor: VCODAPreprocessor
    state_dict: dict[str, Any]
    threshold: float | None
    anomaly_threshold: float | None = None
    domain_adaptation: dict[str, Any] | None = None

    def create_model(self, device: str = "cpu") -> Any:
        torch = _torch()
        model = build_network(self.architecture, self.input_dim, len(self.classes), self.profile, self.sequence_length)
        model.load_state_dict(self.state_dict)
        model.to(device).eval()
        return model

    def predict_proba(self, frame: pd.DataFrame, device: str = "cpu") -> np.ndarray:
        if self.architecture in {"cnn1d", "cnn_lstm", "transformer"}:
            raise ValueError("Temporal deep models require sequence inference via predict_sequences")
        torch = _torch()
        x = self.preprocessor.transform(frame)
        model = self.create_model(device)
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            logits = model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict_sequences(self, sequence_array: np.ndarray, device: str = "cpu") -> np.ndarray:
        torch = _torch()
        model = self.create_model(device)
        with torch.no_grad():
            tensor = torch.as_tensor(sequence_array, dtype=torch.float32, device=device)
            return torch.softmax(model(tensor), dim=1).cpu().numpy()

    def predict_proba_mc(self, frame: pd.DataFrame, device: str = "cpu", samples: int = 20) -> dict[str, np.ndarray]:
        """MC-Dropout predictive mean/std: epistemic uncertainty from stochastic
        forward passes, instead of a single deterministic probability estimate."""
        if self.architecture in {"cnn1d", "cnn_lstm", "transformer"}:
            raise ValueError("Temporal deep models require sequence inference via predict_sequences_mc")
        torch = _torch()
        x = self.preprocessor.transform(frame)
        model = self.create_model(device)
        enable_mc_dropout(model)
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        outputs = []
        with torch.no_grad():
            for _ in range(max(samples, 1)):
                outputs.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
        stacked = np.stack(outputs, axis=0)
        return {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}

    def predict_sequences_mc(self, sequence_array: np.ndarray, device: str = "cpu", samples: int = 20) -> dict[str, np.ndarray]:
        torch = _torch()
        model = self.create_model(device)
        enable_mc_dropout(model)
        tensor = torch.as_tensor(sequence_array, dtype=torch.float32, device=device)
        outputs = []
        with torch.no_grad():
            for _ in range(max(samples, 1)):
                outputs.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
        stacked = np.stack(outputs, axis=0)
        return {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}

    def reconstruction_error(self, frame: pd.DataFrame, device: str = "cpu") -> np.ndarray:
        if self.architecture != "autoencoder":
            raise ValueError("Only autoencoder bundles produce reconstruction errors")
        torch = _torch()
        x = self.preprocessor.transform(frame)
        model = self.create_model(device)
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            reconstructed = model(tensor)
            return ((tensor - reconstructed) ** 2).mean(dim=1).cpu().numpy()

    def save(self, path: str | Path) -> dict[str, str]:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return {"path": str(target), "sha256": sha256_file(target)}

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "DeepBundle":
        target = resolve_path(path)
        if expected_sha256 and sha256_file(target) != expected_sha256:
            raise ValueError("Deep model checksum mismatch")
        value = joblib.load(target)
        if not isinstance(value, cls):
            raise TypeError("Invalid deep model bundle")
        return value


@dataclass
class MultiTaskBundle:
    """Shared-trunk model with a binary head and a multiclass head trained jointly,
    so both objectives regularise the same learned representation."""

    classes_binary: list[str]
    classes_multiclass: list[str]
    input_dim: int
    profile: str
    preprocessor: VCODAPreprocessor
    state_dict: dict[str, Any]
    binary_threshold: float | None
    domain_adaptation: dict[str, Any] | None = None

    def create_model(self, device: str = "cpu") -> Any:
        model = build_multitask_network(self.input_dim, len(self.classes_binary), len(self.classes_multiclass), self.profile)
        model.load_state_dict(self.state_dict)
        model.to(device).eval()
        return model

    def predict_proba(self, frame: pd.DataFrame, device: str = "cpu") -> dict[str, np.ndarray]:
        torch = _torch()
        x = self.preprocessor.transform(frame)
        model = self.create_model(device)
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            binary_logits, multiclass_logits = model(tensor)
            return {
                "binary": torch.softmax(binary_logits, dim=1).cpu().numpy(),
                "multiclass": torch.softmax(multiclass_logits, dim=1).cpu().numpy(),
            }

    def save(self, path: str | Path) -> dict[str, str]:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return {"path": str(target), "sha256": sha256_file(target)}

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str | None = None) -> "MultiTaskBundle":
        target = resolve_path(path)
        if expected_sha256 and sha256_file(target) != expected_sha256:
            raise ValueError("Multitask model checksum mismatch")
        value = joblib.load(target)
        if not isinstance(value, cls):
            raise TypeError("Invalid multitask model bundle")
        return value


def train_deep(
    architecture: str,
    config_path: str | Path = "configs/training.yaml",
    *,
    task: str = "binary",
    promote: bool = True,
    hyperparameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch = _torch()
    config = load_yaml(config_path)
    cfg = dict(config["deep_learning"])
    cfg.update(hyperparameter_overrides or {})
    profile = str(cfg.get("profile", "medium"))
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_rows = config["supervised"].get("max_training_rows", 1_500_000)
    train_frame = load_split("train", config["dataset"]["prepared_dir"], max_rows=max_rows)
    validation_frame = load_split("validation", config["dataset"]["prepared_dir"], max_rows=max_rows)
    test_frame = load_split("test", config["dataset"]["prepared_dir"], max_rows=max_rows)
    target = "binary_label" if task == "binary" else "attack_label"
    x_train_df, y_train_raw, train_meta = split_xy(train_frame, target)
    x_validation_df, y_validation_raw, validation_meta = split_xy(validation_frame, target)
    x_test_df, y_test_raw, test_meta = split_xy(test_frame, target)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    if task == "binary":
        classes = ["benign", "attack"]
        y_train = y_train_raw.astype(int).to_numpy()
        y_validation = y_validation_raw.astype(int).to_numpy()
        y_test = y_test_raw.astype(int).to_numpy()
    else:
        encoder.fit(y_train_raw.astype(str))
        classes = encoder.classes_.tolist()
        known = set(classes)

        val_mask = y_validation_raw.astype(str).isin(known)
        x_validation_df, validation_meta, y_validation_raw = x_validation_df.loc[val_mask].reset_index(drop=True), validation_meta.loc[val_mask].reset_index(drop=True), y_validation_raw.loc[val_mask].reset_index(drop=True)

        known_mask = y_test_raw.astype(str).isin(known)
        x_test_df, test_meta, y_test_raw = x_test_df.loc[known_mask].reset_index(drop=True), test_meta.loc[known_mask].reset_index(drop=True), y_test_raw.loc[known_mask].reset_index(drop=True)

        y_train = encoder.transform(y_train_raw.astype(str))
        y_validation = encoder.transform(y_validation_raw.astype(str))
        y_test = encoder.transform(y_test_raw.astype(str))

    preprocessor = VCODAPreprocessor(scaler=config["preprocessing"].get("numeric_scaler", "robust"))
    x_train = preprocessor.fit_transform(x_train_df)
    x_validation = preprocessor.transform(x_validation_df)
    x_test = preprocessor.transform(x_test_df)

    domain_cfg = cfg.get("domain_adaptation", {}) or {}
    domain_adaptation_meta: dict[str, Any] | None = None
    if bool(domain_cfg.get("enabled", False)) and str(domain_cfg.get("method", "coral")).lower() == "coral":
        # Unsupervised domain adaptation: align the labeled train-domain features onto the
        # *unlabeled* validation+test feature distribution (CORAL, Sun & Saenko 2016). No
        # target labels are used, so this does not leak test ground truth — only the fact
        # that deployment traffic has a different feature distribution than the training
        # capture, which is legitimate prior knowledge in a domain-adaptation setting.
        target_features = np.concatenate([x_validation, x_test], axis=0)
        alignment = coral_fit(x_train, target_features, eps=float(domain_cfg.get("eps", 1e-3)))
        x_train = coral_apply(x_train, alignment)
        domain_adaptation_meta = {"method": "coral", "eps": float(domain_cfg.get("eps", 1e-3))}

    sequence_length = int(cfg.get("sequence_length", 32))
    sequence_model = architecture in {"cnn1d", "cnn_lstm", "transformer"}
    if sequence_model:
        train_dataset = TemporalSequenceDataset(
            x_train, y_train, train_meta, sequence_length, bool(cfg.get("require_true_sequence_order", True))
        )
        validation_dataset = TemporalSequenceDataset(
            x_validation, y_validation, validation_meta, sequence_length, bool(cfg.get("require_true_sequence_order", True))
        )
        test_dataset = TemporalSequenceDataset(
            x_test, y_test, test_meta, sequence_length, bool(cfg.get("require_true_sequence_order", True))
        )
    else:
        if architecture == "autoencoder":
            benign_mask = y_train == 0
            x_train = x_train[benign_mask]
            y_train = y_train[benign_mask]

        train_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long)
        )
        validation_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x_validation, dtype=torch.float32), torch.as_tensor(y_validation, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.long)
        )

    batch_size = int(cfg.get("batch_size", 1024))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    model = build_network(architecture, x_train.shape[-1], len(classes), profile, sequence_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 0.0001)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if architecture == "autoencoder":
        criterion: Any = torch.nn.MSELoss()
        loss_name = "mean_squared_error"
    else:
        y_train_targets = train_dataset.targets if sequence_model else y_train
        counts = np.bincount(y_train_targets, minlength=len(classes))
        weights = len(y_train_targets) / np.maximum(counts * len(classes), 1)
        class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        configured_loss = str(cfg.get("loss", "class_weighted_cross_entropy")).lower()
        if configured_loss == "focal":
            criterion = FocalLoss(alpha=class_weights, gamma=float(cfg.get("focal_gamma", 2.0)))
            loss_name = "focal"
        elif configured_loss == "class_weighted_cross_entropy":
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss_name = "class_weighted_cross_entropy"
        else:
            raise ValueError(f"Unsupported deep-learning loss: {configured_loss}")
    use_amp = bool(cfg.get("mixed_precision", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    patience = int(cfg.get("early_stopping_patience", 5))
    stale = 0
    history: list[dict[str, float]] = []
    writer: Any | None = None
    if bool(cfg.get("tensorboard", True)) and importlib.util.find_spec("torch.utils.tensorboard") is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(resolve_path(f"artifacts/reports/tensorboard/{architecture}_{task}")))
    for epoch in range(int(cfg.get("epochs", 30))):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(xb)
                loss = criterion(output, xb) if architecture == "autoencoder" else criterion(output, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(xb)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for xb, yb in validation_loader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, xb) if architecture == "autoencoder" else criterion(output, yb)
                validation_loss += float(loss.detach().cpu()) * len(xb)
        train_loss /= max(len(train_dataset), 1)
        validation_loss /= max(len(validation_dataset), 1)
        scheduler.step(validation_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/validation", validation_loss, epoch + 1)
            writer.add_scalar("optimiser/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if writer is not None:
        writer.flush()
        writer.close()
    if best_state is None:
        if len(train_dataset) == 0 or len(validation_dataset) == 0:
            raise RuntimeError("Deep model did not produce a checkpoint because training or validation data is empty")
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    if bool(domain_cfg.get("enabled", False)) and bool(domain_cfg.get("adabn", True)):
        # Test-time adaptation (AdaBN, Li et al. 2018): recompute BatchNorm running
        # statistics from unlabeled target-domain forward passes so normalization
        # matches the deployment distribution, without touching any learned weight.
        if sequence_model:
            # Fresh, unpinned loaders (re-iterating the pin_memory=True training loaders
            # a second time triggered a CUDA "resource already mapped" error), and streamed
            # batch-by-batch rather than concatenated (sequence windows balloon memory by
            # the window length, which OOM'd when materialized as one tensor).
            adabn_validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            adabn_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            def _chained_batches() -> Any:
                yield from (xb for xb, _ in adabn_validation_loader)
                yield from (xb for xb, _ in adabn_test_loader)

            batches: Any = _chained_batches()
        else:
            target_tensor = torch.as_tensor(np.concatenate([x_validation, x_test], axis=0), dtype=torch.float32)
            batches = torch.split(target_tensor, batch_size)
        if recalibrate_batchnorm(model, batches, device):
            domain_adaptation_meta = {**(domain_adaptation_meta or {}), "adabn": True}

    threshold: float | None = None
    anomaly_threshold: float | None = None
    if architecture == "autoencoder":
        with torch.no_grad():
            validation_tensor = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
            errors = ((validation_tensor - model(validation_tensor)) ** 2).mean(dim=1).cpu().numpy()
        benign_errors = errors[y_validation == 0]
        anomaly_threshold = float(np.quantile(benign_errors, 1 - float(config["anomaly"].get("target_benign_fpr", 0.01))))
        test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        with torch.no_grad():
            test_errors = ((test_tensor - model(test_tensor)) ** 2).mean(dim=1).cpu().numpy()
        scale = max(float(np.quantile(benign_errors, 0.999)), anomaly_threshold, 1e-9)
        anomaly_probability = np.clip(test_errors / scale, 0, 1)
        probabilities = np.column_stack([1 - anomaly_probability, anomaly_probability])
        metrics = evaluate_classification(y_test, probabilities, classes=["benign", "anomaly"], threshold=anomaly_threshold / scale)
        metrics["anomaly_threshold"] = anomaly_threshold
        validation_prediction_frame = validation_meta.copy()
        validation_prediction_frame["actual"] = y_validation
        validation_prediction_frame["probability_attack"] = np.clip(errors / scale, 0, 1)
        validation_prediction_frame["prediction"] = (errors >= anomaly_threshold).astype(int)
        validation_prediction_frame.to_parquet(resolve_path(f"artifacts/reports/deep_{architecture}_{task}_validation_predictions.parquet"), index=False)
        test_prediction_frame = test_meta.copy()
        test_prediction_frame["actual"] = y_test
        test_prediction_frame["probability_attack"] = anomaly_probability
        test_prediction_frame["prediction"] = (test_errors >= anomaly_threshold).astype(int)
        test_prediction_frame.to_parquet(resolve_path(f"artifacts/reports/deep_{architecture}_{task}_test_predictions.parquet"), index=False)
    else:
        if sequence_model:
            validation_probabilities = []
            validation_targets = []
            with torch.no_grad():
                for xb, yb in validation_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output = torch.softmax(model(xb), dim=1)
                    validation_probabilities.append(output.cpu().numpy())
                    validation_targets.append(yb.cpu().numpy())
                probabilities = []
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output = torch.softmax(model(xb), dim=1)
                    probabilities.append(output.cpu().numpy())
            validation_probabilities = np.vstack(validation_probabilities) if validation_probabilities else np.empty((0, len(classes)))
            validation_targets = np.concatenate(validation_targets) if validation_targets else np.empty((0,), dtype=np.int64)
            probabilities = np.vstack(probabilities) if probabilities else np.empty((0, len(classes)))
            validation_probabilities = np.clip(np.nan_to_num(validation_probabilities, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            probabilities = np.clip(np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            threshold = choose_binary_threshold(validation_targets, validation_probabilities[:, 1]) if task == "binary" else None
            test_targets = test_dataset.targets
            metrics = evaluate_classification(test_targets, probabilities, classes=classes, threshold=threshold or 0.5)
            validation_prediction_frame = validation_dataset.sequence_metadata.copy()
            validation_prediction_frame["actual"] = validation_targets
            validation_prediction_frame["prediction"] = (validation_probabilities[:, 1] >= (threshold or 0.5)).astype(int) if task == "binary" else validation_probabilities.argmax(axis=1)
            for index, class_name in enumerate(classes):
                validation_prediction_frame[f"probability_{class_name}"] = validation_probabilities[:, index]
            validation_prediction_frame.to_parquet(resolve_path(f"artifacts/reports/deep_{architecture}_{task}_validation_predictions.parquet"), index=False)
            test_prediction_frame = test_dataset.sequence_metadata.copy()
            test_prediction_frame["actual"] = test_targets
            test_prediction_frame["prediction"] = (probabilities[:, 1] >= (threshold or 0.5)).astype(int) if task == "binary" else probabilities.argmax(axis=1)
            for index, class_name in enumerate(classes):
                test_prediction_frame[f"probability_{class_name}"] = probabilities[:, index]
            test_prediction_frame.to_parquet(resolve_path(f"artifacts/reports/deep_{architecture}_{task}_test_predictions.parquet"), index=False)
        else:
            test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
            validation_tensor = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
            with torch.no_grad():
                validation_probabilities = torch.softmax(model(validation_tensor), dim=1).cpu().numpy()
                validation_probabilities = np.nan_to_num(validation_probabilities, nan=0.0, posinf=1.0, neginf=0.0)
                validation_probabilities = np.clip(validation_probabilities, 0.0, 1.0)
                probabilities = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
                probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
                probabilities = np.clip(probabilities, 0.0, 1.0)
            threshold = choose_binary_threshold(y_validation, validation_probabilities[:, 1]) if task == "binary" else None
            metrics = evaluate_classification(y_test, probabilities, classes=classes, threshold=threshold or 0.5)
            validation_prediction_frame = validation_meta.copy()
            validation_prediction_frame["actual"] = y_validation
            validation_prediction_frame["prediction"] = (validation_probabilities[:, 1] >= (threshold or 0.5)).astype(int) if task == "binary" else validation_probabilities.argmax(axis=1)
            for index, class_name in enumerate(classes):
                validation_prediction_frame[f"probability_{class_name}"] = validation_probabilities[:, index]
            validation_prediction_frame.to_parquet(resolve_path(f"artifacts/reports/deep_{architecture}_{task}_validation_predictions.parquet"), index=False)
            test_prediction_frame = test_meta.copy()
            test_prediction_frame["actual"] = y_test
            test_prediction_frame["prediction"] = (probabilities[:, 1] >= (threshold or 0.5)).astype(int) if task == "binary" else probabilities.argmax(axis=1)
            for index, class_name in enumerate(classes):
                test_prediction_frame[f"probability_{class_name}"] = probabilities[:, index]
            test_prediction_frame.to_parquet(resolve_path(f"artifacts/reports/deep_{architecture}_{task}_test_predictions.parquet"), index=False)

    bundle = DeepBundle(
        architecture=architecture, task="anomaly" if architecture == "autoencoder" else task,
        classes=[str(c) for c in classes], input_dim=x_train.shape[-1], profile=profile,
        sequence_length=sequence_length, preprocessor=preprocessor, state_dict=best_state,
        threshold=threshold, anomaly_threshold=anomaly_threshold, domain_adaptation=domain_adaptation_meta,
    )
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = resolve_path(f"models/deep_learning/{architecture}_{task}_{version}.joblib")
    bundle.save(staging)
    manifest = load_manifest(config["dataset"]["prepared_dir"])
    registry_task = "anomaly_autoencoder" if architecture == "autoencoder" else f"deep_{task}"
    registry = ModelRegistry()
    record = registry.register(
        model_name=f"{architecture}_{task}", version=version, task=registry_task,
        artifact_path=staging, metrics=metrics, dataset_fingerprint=manifest["dataset_fingerprint"],
        feature_list=preprocessor.schema.all_features if preprocessor.schema else [], preprocessing_version="1.0.0",
        hyperparameters={
            "architecture": architecture, "profile": profile, "epochs_run": len(history),
            "device": device, "loss": loss_name, "mixed_precision": use_amp,
            "domain_adaptation": domain_adaptation_meta,
            "hyperparameter_overrides": hyperparameter_overrides,
        },
        threshold=threshold if threshold is not None else anomaly_threshold,
    )
    if promote:
        registry.promote(record["model_name"], record["version"])
    report = {
        "architecture": architecture, "task": task, "device": device, "loss": loss_name,
        "mixed_precision": use_amp, "history": history, "metrics": metrics,
        "registry_record": record,
    }
    report_path = f"artifacts/reports/deep_{architecture}_{task}_report.json"
    dump_json(report, report_path)
    tracker = ExperimentTracker(f"deep_{architecture}_{task}", backend="local")
    tracker.log_parameters({"architecture": architecture, "task": task, "profile": profile, "device": device, "epochs_run": len(history)})
    tracker.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
    tracker.log_artifact(report_path)
    tracker.log_artifact(staging)
    tracker.finish()
    return report


def search_deep_hyperparameters(
    architecture: str,
    config_path: str | Path = "configs/training.yaml",
    *,
    task: str = "binary",
    trials: int = 6,
    quick_epochs: int = 3,
    max_rows: int = 150_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Optuna TPE search over (learning_rate, weight_decay, gradient_clip) using short,
    subsampled training runs as a cheap proxy for the full training objective. Mirrors
    the Optuna pattern already used for the tree-based supervised models, extended to
    the deep-learning architectures, which previously trained on fixed hyperparameters."""
    if trials <= 0 or importlib.util.find_spec("optuna") is None:
        return {}
    import optuna

    torch = _torch()
    config = load_yaml(config_path)
    cfg = dict(config["deep_learning"])
    profile = str(cfg.get("profile", "medium"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_frame = load_split("train", config["dataset"]["prepared_dir"], max_rows=max_rows)
    validation_frame = load_split("validation", config["dataset"]["prepared_dir"], max_rows=max_rows)
    target = "binary_label" if task == "binary" else "attack_label"
    x_train_df, y_train_raw, train_meta = split_xy(train_frame, target)
    x_validation_df, y_validation_raw, validation_meta = split_xy(validation_frame, target)

    if task == "binary":
        classes: list[Any] = ["benign", "attack"]
        y_train = y_train_raw.astype(int).to_numpy()
        y_validation = y_validation_raw.astype(int).to_numpy()
    else:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder().fit(y_train_raw.astype(str))
        classes = encoder.classes_.tolist()
        known_mask = y_validation_raw.astype(str).isin(set(classes))
        x_validation_df = x_validation_df.loc[known_mask].reset_index(drop=True)
        validation_meta = validation_meta.loc[known_mask].reset_index(drop=True)
        y_validation_raw = y_validation_raw.loc[known_mask].reset_index(drop=True)
        y_train = encoder.transform(y_train_raw.astype(str))
        y_validation = encoder.transform(y_validation_raw.astype(str))

    preprocessor = VCODAPreprocessor(scaler=config["preprocessing"].get("numeric_scaler", "robust"))
    x_train = preprocessor.fit_transform(x_train_df)
    x_validation = preprocessor.transform(x_validation_df)
    sequence_length = int(cfg.get("sequence_length", 32))
    sequence_model = architecture in {"cnn1d", "cnn_lstm", "transformer"}

    if sequence_model:
        train_dataset: Any = TemporalSequenceDataset(x_train, y_train, train_meta, sequence_length, bool(cfg.get("require_true_sequence_order", True)))
        validation_dataset: Any = TemporalSequenceDataset(x_validation, y_validation, validation_meta, sequence_length, bool(cfg.get("require_true_sequence_order", True)))
    elif architecture == "autoencoder":
        benign_mask = y_train == 0
        train_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x_train[benign_mask], dtype=torch.float32), torch.as_tensor(y_train[benign_mask], dtype=torch.long)
        )
        validation_dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x_validation, dtype=torch.float32), torch.as_tensor(y_validation, dtype=torch.long)
        )
    else:
        train_dataset = torch.utils.data.TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
        validation_dataset = torch.utils.data.TensorDataset(torch.as_tensor(x_validation, dtype=torch.float32), torch.as_tensor(y_validation, dtype=torch.long))

    batch_size = int(cfg.get("batch_size", 1024))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    input_dim = x_train.shape[-1]

    def objective(trial: Any) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 2.0),
        }
        torch.manual_seed(seed)
        model = build_network(architecture, input_dim, len(classes), profile, sequence_length).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        criterion: Any = torch.nn.MSELoss() if architecture == "autoencoder" else torch.nn.CrossEntropyLoss()
        for _ in range(max(quick_epochs, 1)):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(xb)
                loss = criterion(output, xb) if architecture == "autoencoder" else criterion(output, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip"])
                optimizer.step()
        model.eval()
        if architecture == "autoencoder":
            total, count = 0.0, 0
            with torch.no_grad():
                for xb, _ in validation_loader:
                    xb = xb.to(device)
                    output = model(xb)
                    total += float(((xb - output) ** 2).mean().cpu()) * len(xb)
                    count += len(xb)
            return -(total / max(count, 1))
        from sklearn.metrics import f1_score
        predictions_list, targets_list = [], []
        with torch.no_grad():
            for xb, yb in validation_loader:
                output = model(xb.to(device))
                predictions_list.append(output.argmax(dim=1).cpu().numpy())
                targets_list.append(yb.numpy())
        predictions = np.concatenate(predictions_list) if predictions_list else np.empty((0,))
        targets = np.concatenate(targets_list) if targets_list else np.empty((0,))
        return float(f1_score(targets, predictions, average="macro", zero_division=0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


def train_deep_multitask(
    config_path: str | Path = "configs/training.yaml",
    *,
    promote: bool = True,
) -> dict[str, Any]:
    """Shared-trunk model jointly optimised for binary attack/benign detection and
    fine-grained attack-family classification. Each objective regularises the same
    learned representation instead of training two disconnected models, and the
    binary head's predictions join the stacking ensemble automatically (they are
    written to the same `deep_*_binary_{validation,test}_predictions.parquet`
    naming convention every other ensemble member uses)."""
    torch = _torch()
    config = load_yaml(config_path)
    cfg = dict(config["deep_learning"])
    multitask_cfg = cfg.get("multitask", {}) or {}
    profile = str(cfg.get("profile", "medium"))
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_rows = config["supervised"].get("max_training_rows", 1_500_000)
    train_frame = load_split("train", config["dataset"]["prepared_dir"], max_rows=max_rows)
    validation_frame = load_split("validation", config["dataset"]["prepared_dir"], max_rows=max_rows)
    test_frame = load_split("test", config["dataset"]["prepared_dir"], max_rows=max_rows)

    x_train_df, y_train_binary_raw, train_meta = split_xy(train_frame, "binary_label")
    x_validation_df, y_validation_binary_raw, validation_meta = split_xy(validation_frame, "binary_label")
    x_test_df, y_test_binary_raw, test_meta = split_xy(test_frame, "binary_label")
    y_train_multiclass_raw = train_frame["attack_label"].copy()
    y_validation_multiclass_raw = validation_frame["attack_label"].copy()
    y_test_multiclass_raw = test_frame["attack_label"].copy()

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder().fit(y_train_multiclass_raw.astype(str))
    classes_multiclass = encoder.classes_.tolist()
    known = set(classes_multiclass)

    y_train_binary = y_train_binary_raw.astype(int).to_numpy()
    y_train_multiclass = encoder.transform(y_train_multiclass_raw.astype(str))

    val_known = y_validation_multiclass_raw.astype(str).isin(known)
    x_validation_df = x_validation_df.loc[val_known].reset_index(drop=True)
    validation_meta = validation_meta.loc[val_known].reset_index(drop=True)
    y_validation_binary = y_validation_binary_raw.loc[val_known].astype(int).to_numpy()
    y_validation_multiclass = encoder.transform(y_validation_multiclass_raw.loc[val_known].astype(str))

    test_known = y_test_multiclass_raw.astype(str).isin(known)
    x_test_df = x_test_df.loc[test_known].reset_index(drop=True)
    test_meta = test_meta.loc[test_known].reset_index(drop=True)
    y_test_binary = y_test_binary_raw.loc[test_known].astype(int).to_numpy()
    y_test_multiclass = encoder.transform(y_test_multiclass_raw.loc[test_known].astype(str))

    preprocessor = VCODAPreprocessor(scaler=config["preprocessing"].get("numeric_scaler", "robust"))
    x_train = preprocessor.fit_transform(x_train_df)
    x_validation = preprocessor.transform(x_validation_df)
    x_test = preprocessor.transform(x_test_df)

    domain_cfg = cfg.get("domain_adaptation", {}) or {}
    domain_adaptation_meta: dict[str, Any] | None = None
    if bool(domain_cfg.get("enabled", False)) and str(domain_cfg.get("method", "coral")).lower() == "coral":
        target_features = np.concatenate([x_validation, x_test], axis=0)
        alignment = coral_fit(x_train, target_features, eps=float(domain_cfg.get("eps", 1e-3)))
        x_train = coral_apply(x_train, alignment)
        domain_adaptation_meta = {"method": "coral", "eps": float(domain_cfg.get("eps", 1e-3))}

    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train_binary, dtype=torch.long),
        torch.as_tensor(y_train_multiclass, dtype=torch.long),
    )
    validation_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_validation, dtype=torch.float32),
        torch.as_tensor(y_validation_binary, dtype=torch.long),
        torch.as_tensor(y_validation_multiclass, dtype=torch.long),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_test, dtype=torch.float32),
        torch.as_tensor(y_test_binary, dtype=torch.long),
        torch.as_tensor(y_test_multiclass, dtype=torch.long),
    )
    batch_size = int(cfg.get("batch_size", 1024))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    model = build_multitask_network(x_train.shape[-1], 2, len(classes_multiclass), profile).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 0.001)), weight_decay=float(cfg.get("weight_decay", 0.0001)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    binary_counts = np.bincount(y_train_binary, minlength=2)
    binary_weights = torch.as_tensor(len(y_train_binary) / np.maximum(binary_counts * 2, 1), dtype=torch.float32, device=device)
    multiclass_counts = np.bincount(y_train_multiclass, minlength=len(classes_multiclass))
    multiclass_weights = torch.as_tensor(
        len(y_train_multiclass) / np.maximum(multiclass_counts * len(classes_multiclass), 1), dtype=torch.float32, device=device
    )
    binary_criterion = torch.nn.CrossEntropyLoss(weight=binary_weights)
    multiclass_criterion = torch.nn.CrossEntropyLoss(weight=multiclass_weights)
    multiclass_loss_weight = float(multitask_cfg.get("multiclass_weight", 1.0))

    use_amp = bool(cfg.get("mixed_precision", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    patience = int(cfg.get("early_stopping_patience", 5))
    stale = 0
    history: list[dict[str, float]] = []
    writer: Any | None = None
    if bool(cfg.get("tensorboard", True)) and importlib.util.find_spec("torch.utils.tensorboard") is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(resolve_path("artifacts/reports/tensorboard/multitask_binary")))

    for epoch in range(int(cfg.get("epochs", 30))):
        model.train()
        train_loss = 0.0
        for xb, yb_binary, yb_multiclass in train_loader:
            xb, yb_binary, yb_multiclass = xb.to(device), yb_binary.to(device), yb_multiclass.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                binary_logits, multiclass_logits = model(xb)
                loss = binary_criterion(binary_logits, yb_binary) + multiclass_loss_weight * multiclass_criterion(multiclass_logits, yb_multiclass)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(xb)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for xb, yb_binary, yb_multiclass in validation_loader:
                xb, yb_binary, yb_multiclass = xb.to(device), yb_binary.to(device), yb_multiclass.to(device)
                binary_logits, multiclass_logits = model(xb)
                loss = binary_criterion(binary_logits, yb_binary) + multiclass_loss_weight * multiclass_criterion(multiclass_logits, yb_multiclass)
                validation_loss += float(loss.detach().cpu()) * len(xb)
        train_loss /= max(len(train_dataset), 1)
        validation_loss /= max(len(validation_dataset), 1)
        scheduler.step(validation_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/validation", validation_loss, epoch + 1)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if writer is not None:
        writer.flush()
        writer.close()
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    if bool(domain_cfg.get("enabled", False)) and bool(domain_cfg.get("adabn", True)):
        target_tensor = torch.as_tensor(np.concatenate([x_validation, x_test], axis=0), dtype=torch.float32)
        if recalibrate_batchnorm(model, torch.split(target_tensor, batch_size), device):
            domain_adaptation_meta = {**(domain_adaptation_meta or {}), "adabn": True}

    with torch.no_grad():
        validation_tensor = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
        binary_logits, multiclass_logits = model(validation_tensor)
        validation_binary_probabilities = np.clip(np.nan_to_num(torch.softmax(binary_logits, dim=1).cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        validation_multiclass_probabilities = np.clip(
            np.nan_to_num(torch.softmax(multiclass_logits, dim=1).cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
        test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        binary_logits, multiclass_logits = model(test_tensor)
        test_binary_probabilities = np.clip(np.nan_to_num(torch.softmax(binary_logits, dim=1).cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        test_multiclass_probabilities = np.clip(
            np.nan_to_num(torch.softmax(multiclass_logits, dim=1).cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )

    binary_threshold = choose_binary_threshold(y_validation_binary, validation_binary_probabilities[:, 1])
    binary_metrics = evaluate_classification(y_test_binary, test_binary_probabilities, classes=["benign", "attack"], threshold=binary_threshold)
    multiclass_metrics = evaluate_classification(y_test_multiclass, test_multiclass_probabilities, classes=classes_multiclass)

    validation_binary_frame = validation_meta.copy()
    validation_binary_frame["actual"] = y_validation_binary
    validation_binary_frame["prediction"] = (validation_binary_probabilities[:, 1] >= binary_threshold).astype(int)
    validation_binary_frame["probability_benign"] = validation_binary_probabilities[:, 0]
    validation_binary_frame["probability_attack"] = validation_binary_probabilities[:, 1]
    validation_binary_frame.to_parquet(resolve_path("artifacts/reports/deep_multitask_binary_validation_predictions.parquet"), index=False)

    test_binary_frame = test_meta.copy()
    test_binary_frame["actual"] = y_test_binary
    test_binary_frame["prediction"] = (test_binary_probabilities[:, 1] >= binary_threshold).astype(int)
    test_binary_frame["probability_benign"] = test_binary_probabilities[:, 0]
    test_binary_frame["probability_attack"] = test_binary_probabilities[:, 1]
    test_binary_frame.to_parquet(resolve_path("artifacts/reports/deep_multitask_binary_test_predictions.parquet"), index=False)

    validation_multiclass_frame = validation_meta.copy()
    validation_multiclass_frame["actual"] = y_validation_multiclass
    validation_multiclass_frame["prediction"] = validation_multiclass_probabilities.argmax(axis=1)
    for index, class_name in enumerate(classes_multiclass):
        validation_multiclass_frame[f"probability_{class_name}"] = validation_multiclass_probabilities[:, index]
    validation_multiclass_frame.to_parquet(resolve_path("artifacts/reports/deep_multitask_multiclass_validation_predictions.parquet"), index=False)

    test_multiclass_frame = test_meta.copy()
    test_multiclass_frame["actual"] = y_test_multiclass
    test_multiclass_frame["prediction"] = test_multiclass_probabilities.argmax(axis=1)
    for index, class_name in enumerate(classes_multiclass):
        test_multiclass_frame[f"probability_{class_name}"] = test_multiclass_probabilities[:, index]
    test_multiclass_frame.to_parquet(resolve_path("artifacts/reports/deep_multitask_multiclass_test_predictions.parquet"), index=False)

    bundle = MultiTaskBundle(
        classes_binary=["benign", "attack"], classes_multiclass=[str(c) for c in classes_multiclass],
        input_dim=x_train.shape[-1], profile=profile, preprocessor=preprocessor, state_dict=best_state,
        binary_threshold=binary_threshold, domain_adaptation=domain_adaptation_meta,
    )
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = resolve_path(f"models/deep_learning/multitask_{version}.joblib")
    bundle.save(staging)
    manifest = load_manifest(config["dataset"]["prepared_dir"])
    registry = ModelRegistry()
    binary_record = registry.register(
        model_name="multitask_binary", version=version, task="deep_multitask_binary",
        artifact_path=staging, metrics=binary_metrics, dataset_fingerprint=manifest["dataset_fingerprint"],
        feature_list=preprocessor.schema.all_features if preprocessor.schema else [], preprocessing_version="1.0.0",
        hyperparameters={
            "architecture": "multitask", "profile": profile, "epochs_run": len(history), "device": device,
            "multiclass_weight": multiclass_loss_weight, "domain_adaptation": domain_adaptation_meta,
        },
        threshold=binary_threshold,
    )
    multiclass_record = registry.register(
        model_name="multitask_multiclass", version=version, task="deep_multitask_multiclass",
        artifact_path=staging, metrics=multiclass_metrics, dataset_fingerprint=manifest["dataset_fingerprint"],
        feature_list=preprocessor.schema.all_features if preprocessor.schema else [], preprocessing_version="1.0.0",
        hyperparameters={
            "architecture": "multitask", "profile": profile, "epochs_run": len(history), "device": device,
            "multiclass_weight": multiclass_loss_weight, "domain_adaptation": domain_adaptation_meta,
        },
        threshold=None,
    )
    if promote:
        registry.promote(binary_record["model_name"], binary_record["version"])
        registry.promote(multiclass_record["model_name"], multiclass_record["version"])
    report = {
        "architecture": "multitask", "device": device, "history": history,
        "binary_metrics": binary_metrics, "multiclass_metrics": multiclass_metrics,
        "binary_registry_record": binary_record, "multiclass_registry_record": multiclass_record,
    }
    report_path = "artifacts/reports/deep_multitask_report.json"
    dump_json(report, report_path)
    tracker = ExperimentTracker("deep_multitask", backend="local")
    tracker.log_parameters({"architecture": "multitask", "device": device, "epochs_run": len(history)})
    tracker.log_metrics({key: value for key, value in binary_metrics.items() if isinstance(value, (int, float))})
    tracker.log_artifact(report_path)
    tracker.log_artifact(staging)
    tracker.finish()
    return report
