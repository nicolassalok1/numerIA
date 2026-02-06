param(
    [ValidateSet("run", "sub")]
    [string]$Action = "run",
    [string]$ModelId
)

$ErrorActionPreference = "Stop"

# Default: enable progress bars unless explicitly disabled by caller
if (-not $env:NUMERAI_PROGRESS) { $env:NUMERAI_PROGRESS = "1" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$runScript = Join-Path $ScriptDir "run_me.ps1"
$subScript = Join-Path $ScriptDir "sub_me.ps1"

# Load local keys (if present) so NUMERAI_MODEL_ID is available
$keysCandidates = @(
    (Join-Path $ScriptDir "keys_local.ps1"),
    (Join-Path $ScriptDir "numerai-project\\keys_local.ps1"),
    (Join-Path $ScriptDir "numerai-project\\scripts\\keys_local.ps1")
)
foreach ($candidate in $keysCandidates) {
    if (Test-Path $candidate) {
        Write-Host "Loading Numerai credentials from $candidate"
        . $candidate
        break
    }
}

if ($ModelId) {
    $env:NUMERAI_MODEL_ID = $ModelId
    Write-Host "Using NUMERAI_MODEL_ID override: $ModelId"
}
elseif ($env:NUMERAI_LOCAL_MODEL_ID) {
    $env:NUMERAI_MODEL_ID = $env:NUMERAI_LOCAL_MODEL_ID
    Write-Host "Using NUMERAI_LOCAL_MODEL_ID: $($env:NUMERAI_LOCAL_MODEL_ID)"
}

if ($Action -eq "run") {
    if (-not (Test-Path $runScript)) {
        Write-Error "Missing run_me.ps1 at $runScript"
        exit 1
    }
    Write-Host "Running full pipeline via run_me.ps1..."
    & $runScript @args
    exit $LASTEXITCODE
}
else {
    if (-not (Test-Path $subScript)) {
        Write-Error "Missing sub_me.ps1 at $subScript"
        exit 1
    }
    Write-Host "Submitting via sub_me.ps1..."
    & $subScript @args
    exit $LASTEXITCODE
}
