$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# UTF-8 output + proxy cleanup
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""

# Model names (must match nodes.json / Numerai dashboard)
$ClassicModel = "tgrv2"
$SignalsModel = "tgr_sig"
$CryptoModel  = "tgr_cry"

# Locate numerai CLI
$numeraiCmd = Get-Command numerai -ErrorAction SilentlyContinue
$numeraiExe = $null
if ($numeraiCmd) {
    $numeraiExe = $numeraiCmd.Source
} else {
    $fallback = "C:\Users\nicol\AppData\Roaming\Python\Python313\Scripts\numerai.exe"
    if (Test-Path $fallback) { $numeraiExe = $fallback }
}
if (-not $numeraiExe) {
    Write-Error "numerai CLI not found in PATH or fallback path."
    exit 1
}

$failed = $false

Write-Host "=== Health Check: Classic (t8) ==="
& $numeraiExe node -m $ClassicModel -t 8 test -v
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Host "=== Health Check: Signals (t11) ==="
& $numeraiExe node -m $SignalsModel -t 11 test -v
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Host "=== Health Check: Crypto (t12) ==="
& $numeraiExe node -m $CryptoModel -t 12 test -v
if ($LASTEXITCODE -ne 0) { $failed = $true }

if ($failed) {
    Write-Error "One or more nodes failed health check."
    exit 1
}

Write-Host "All nodes healthy."
