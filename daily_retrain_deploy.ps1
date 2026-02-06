$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Optional: force UTF-8 output (helps with CLI logs)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""

# Model names (Numerai dashboard)
$ClassicModel = "salok1"
$SignalsModel = "salok1_signals"
$CryptoModel  = "salok1"

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

Write-Host "=== Classic: train + pickle ==="
& (Join-Path $ScriptDir "run_classic_v5.ps1")

Write-Host "=== Signals: train + pickle ==="
& (Join-Path $ScriptDir "run_signals_v2.ps1")

Write-Host "=== Crypto: train + pickle ==="
& (Join-Path $ScriptDir "run_crypto_v2.ps1")

if (-not $env:NUMERAI_SKIP_DEPLOY) {
    Write-Host "=== Deploy Classic node ==="
    & $numeraiExe node -m $ClassicModel -t 8 deploy -v

    Write-Host "=== Deploy Signals node ==="
    & $numeraiExe node -m $SignalsModel -t 11 deploy -v

    Write-Host "=== Deploy Crypto node ==="
    & $numeraiExe node -m $CryptoModel -t 12 deploy -v
}

if ($env:NUMERAI_DAILY_TEST -eq "1") {
    Write-Host "=== Test Classic node ==="
    & $numeraiExe node -m $ClassicModel -t 8 test -v

    Write-Host "=== Test Signals node ==="
    & $numeraiExe node -m $SignalsModel -t 11 test -v

    Write-Host "=== Test Crypto node ==="
    & $numeraiExe node -m $CryptoModel -t 12 test -v
}
