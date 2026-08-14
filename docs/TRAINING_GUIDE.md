# Training Guide

## 1. Supervised binary model

```powershell
vcoda train supervised --task binary
```

The primary candidate is XGBoost. LightGBM, CatBoost, Random Forest, and Extra Trees are compared where installed. Selection uses configured macro F1 rather than accuracy alone.

Faster first run:

```powershell
vcoda train supervised --task binary --optuna-trials 0
```

Specific candidates:

```powershell
vcoda train supervised --task binary --models "xgboost,extra_trees" --optuna-trials 10
```

## 2. Multiclass attack model

```powershell
vcoda train supervised --task multiclass
```

Attack labels that are absent from training/validation are excluded from final multiclass testing and reported rather than encoded incorrectly.

## 3. Offline anomaly models

```powershell
vcoda train anomaly
```

By default this trains Isolation Forest on benign training traffic and tunes its threshold on validation data under the target benign false-positive rate.

Optional comparisons:

```powershell
vcoda train anomaly --models "isolation_forest,local_outlier_factor,one_class_svm"
```

## 4. Deep per-flow model

```powershell
vcoda train deep --architecture mlp --task binary
```

The PyTorch pipeline supports GPU/CPU, batches, class-weighted loss, early stopping, scheduling, gradient clipping, checkpoint selection, and mixed precision on CUDA.

## 5. Neural autoencoder

```powershell
vcoda train deep --architecture autoencoder --task binary
```

Only benign training rows are used. The anomaly threshold comes from the benign validation error distribution.

## 6. Temporal models

Only run after verifying sequence metadata:

```powershell
vcoda train deep --architecture cnn1d --task binary
vcoda train deep --architecture cnn_lstm --task binary
vcoda train deep --architecture transformer --task binary
```

The models use sequences of length configured in `deep_learning.sequence_length`. They do not treat unrelated feature columns as fake time steps.

## 7. Ensemble

```powershell
vcoda optimise-ensemble
```

It aligns validation predictions using `row_id`, fits a stacking model or searches constrained soft-voting weights, then evaluates once using aligned held-out test predictions.

## 8. Registry

```powershell
vcoda list-models
vcoda promote-model <model-name> <version>
vcoda rollback-model supervised_binary
```

Training never overwrites an active registry version. Each record stores checksum, dataset fingerprint, features, metrics, threshold, and environment metadata.

## Generated reports

```text
artifacts/reports/supervised_binary_training_report.json
artifacts/reports/supervised_multiclass_training_report.json
artifacts/reports/anomaly_training_report.json
artifacts/reports/deep_<architecture>_<task>_report.json
artifacts/reports/ensemble_report.json
models/registry/index.json
models/registry/active.json
```

## Deep-learning loss and TensorBoard

The default neural loss is class-weighted cross-entropy. To test focal loss, edit
`configs/training.yaml`:

```yaml
deep_learning:
  loss: focal
  focal_gamma: 2.0
```

TensorBoard logging is enabled by default and writes under:

```text
artifacts\reports\tensorboard\
```

Launch it with:

```powershell
tensorboard --logdir ".\artifacts\reports\tensorboard"
```
