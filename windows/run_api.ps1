$ErrorActionPreference="Stop"; Set-Location (Split-Path $PSScriptRoot -Parent); & .\.venv\Scripts\vcoda.exe serve-api
