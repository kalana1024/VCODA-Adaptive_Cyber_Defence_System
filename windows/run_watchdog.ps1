$ErrorActionPreference="Stop"; Set-Location (Split-Path $PSScriptRoot -Parent); & .\.venv\Scripts\vcoda.exe heal --watch
