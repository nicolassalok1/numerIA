param(
    [switch]$SkipConfig,
    [switch]$SkipDeploy,
    [switch]$SkipTest,
    [string]$ScheduleTime
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# UTF-8 + proxy cleanup (helps with numerai CLI logs)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""

# Model names + IDs (must match nodes.json / Numerai dashboard)
$ClassicModel   = "tgrv2"
$ClassicModelId = "f35f60cf-6784-45e5-a10c-2f348b6bb2a5"
$SignalsModel   = "tgr_sig"
$SignalsModelId = "d19eb070-251a-4fbd-8acf-60893d6848d0"
$CryptoModel    = "tgr_cry"
$CryptoModelId  = "c574308c-e4d7-4849-886b-aa7377560791"

$ClassicNodeDir = Join-Path $ScriptDir "numerai-project\tgrv2-node"
$SignalsNodeDir = Join-Path $ScriptDir "numerai-project\signals-node"
$CryptoNodeDir  = Join-Path $ScriptDir "numerai-project\crypto-node"

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

function Get-ConfiguredNodeName {
    param([string]$ModelId)
    $nodesPath = Join-Path $env:USERPROFILE ".numerai\nodes.json"
    if (-not (Test-Path $nodesPath)) { return $null }
    try {
        $data = Get-Content -Raw -Path $nodesPath | ConvertFrom-Json
    } catch {
        return $null
    }
    foreach ($prop in $data.PSObject.Properties) {
        if ($prop.Value.model_id -eq $ModelId) {
            return $prop.Name
        }
    }
    return $null
}

function Ensure-NodeConfig {
    param(
        [string]$ModelName,
        [string]$ModelId,
        [int]$Tournament,
        [string]$NodeDir,
        [string]$ExampleEnv = ""
    )
    $existing = Get-ConfiguredNodeName -ModelId $ModelId
    if ($existing) {
        Write-Host "Node already configured for model_id $ModelId ($existing)"
        return
    }
    Write-Host "Configuring node for $ModelName (tournament $Tournament)"
    if ($ExampleEnv) {
        & $numeraiExe node -m $ModelName -t $Tournament config -P aws -s mem-md -e $ExampleEnv -p $NodeDir -v
    } else {
        & $numeraiExe node -m $ModelName -t $Tournament config -P aws -s mem-md -p $NodeDir -v
    }
}

if (-not $SkipConfig) {
    Ensure-NodeConfig -ModelName $ClassicModel -ModelId $ClassicModelId -Tournament 8 -NodeDir $ClassicNodeDir
    Ensure-NodeConfig -ModelName $SignalsModel -ModelId $SignalsModelId -Tournament 11 -NodeDir $SignalsNodeDir
    Ensure-NodeConfig -ModelName $CryptoModel -ModelId $CryptoModelId -Tournament 12 -NodeDir $CryptoNodeDir
}

Write-Host "=== Classic: train + pickle ==="
& (Join-Path $ScriptDir "run_classic_v5.ps1")

Write-Host "=== Signals: train + pickle ==="
& (Join-Path $ScriptDir "run_signals_v2.ps1")

Write-Host "=== Crypto: train + pickle ==="
& (Join-Path $ScriptDir "run_crypto_v2.ps1")

if (-not $SkipDeploy) {
    Write-Host "=== Deploy Classic (TGRV2) node ==="
    & $numeraiExe node -m $ClassicModel -t 8 deploy -v

    Write-Host "=== Deploy Signals node ==="
    & $numeraiExe node -m $SignalsModel -t 11 deploy -v

    Write-Host "=== Deploy Crypto node ==="
    & $numeraiExe node -m $CryptoModel -t 12 deploy -v
}

if (-not $SkipTest) {
    Write-Host "=== Test Classic (TGRV2) node ==="
    & $numeraiExe node -m $ClassicModel -t 8 test -v

    Write-Host "=== Test Signals node ==="
    & $numeraiExe node -m $SignalsModel -t 11 test -v

    Write-Host "=== Test Crypto node ==="
    & $numeraiExe node -m $CryptoModel -t 12 test -v
}

if ($ScheduleTime) {
    Write-Host "Creating scheduled task: Numerai Daily Retrain at $ScheduleTime"
    $taskName = "Numerai Daily Retrain"
    $taskCmd = "pwsh -NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -SkipConfig -SkipTest"
    schtasks /Create /TN $taskName /TR $taskCmd /SC DAILY /ST $ScheduleTime /F | Out-Null
}
