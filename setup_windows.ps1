param(
    [ValidateSet("base", "training", "full")][string]$Profile = "full",
    [switch]$Gpu,
    [string]$CudaIndexUrl = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "V-CODA Windows setup" -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11 or 3.12 is required. Install 64-bit Python and enable PATH."
}
$Python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $Python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,14), sys.version"
if (-not (Test-Path .venv)) { & $Python -m venv .venv }
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -e .
if ($Profile -in @("training", "full")) { & $VenvPython -m pip install -r requirements\training.txt }
if ($Profile -eq "full") {
    if ($Gpu -and $CudaIndexUrl) {
        & $VenvPython -m pip install torch torchvision --index-url $CudaIndexUrl
    } else {
        & $VenvPython -m pip install torch tensorboard
    }
    & $VenvPython -m pip install fastapi "uvicorn[standard]" python-multipart streamlit plotly river scapy pyshark pytest pytest-cov httpx ruff
    & $VenvPython -m pip install -r requirements\cti.txt
}
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& $VenvPython -m vcoda.cli system-check
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Npcap is required only for live capture. TShark/Zeek is optional for additional PCAP extraction." -ForegroundColor Yellow
