param([switch]$SkipOptuna, [switch]$TrainTemporal)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$VCODA = ".\.venv\Scripts\vcoda.exe"
& $VCODA system-check
& $VCODA inspect-data --data-dir ".\data\raw\nf_uq_nids_v2"
& $VCODA prepare-data
$Trials = if ($SkipOptuna) { 0 } else { 25 }
& $VCODA train supervised --task binary --optuna-trials $Trials
& $VCODA train supervised --task multiclass --optuna-trials $Trials
& $VCODA train anomaly
& $VCODA train deep --architecture mlp --task binary
& $VCODA train deep --architecture autoencoder --task binary
if ($TrainTemporal) {
    & $VCODA train deep --architecture cnn1d --task binary
    & $VCODA train deep --architecture cnn_lstm --task binary
    & $VCODA train deep --architecture transformer --task binary
}
& $VCODA optimise-ensemble
& $VCODA evaluate
Write-Host "Training workflow complete. Review artifacts\reports." -ForegroundColor Green
