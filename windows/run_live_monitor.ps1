param([Parameter(Mandatory=$true)][string]$Interface)
$ErrorActionPreference="Stop"; Set-Location (Split-Path $PSScriptRoot -Parent); & .\.venv\Scripts\vcoda.exe monitor-live --interface $Interface
