param(
    [ValidateSet("run", "sub")]
    [string]$Action = "run"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runScript = Join-Path $ScriptDir "run_me.ps1"
$subScript = Join-Path $ScriptDir "sub_me.ps1"

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
