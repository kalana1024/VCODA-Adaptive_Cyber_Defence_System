$ErrorActionPreference="Stop"; Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\python.exe -m compileall -q src scripts dashboard
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\vcoda.exe system-check
& .\.venv\Scripts\vcoda.exe verify-audit
Write-Host "Installation verification completed." -ForegroundColor Green
