$ErrorActionPreference = "Stop"

# Conda activation (required)
$targetCondaEnv = $env:NUMERAI_CONDA_ENV
if (-not $targetCondaEnv) { $targetCondaEnv = "lgbm-gpu" }
$condaActivated = $false
if ($env:CONDA_DEFAULT_ENV -eq $targetCondaEnv) {
    Write-Host "Conda env already active: $targetCondaEnv"
    $condaActivated = $true
}
else {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        Write-Error "conda introuvable dans PATH; active l'environnement '$targetCondaEnv' avant de lancer."
        exit 1
    }
    try {
        (& $condaCmd "shell.powershell" "hook") | Out-String | Invoke-Expression
        conda activate $targetCondaEnv | Out-Null
        Write-Host "Activated conda env: $targetCondaEnv"
        $condaActivated = $true
    }
    catch {
        Write-Error "Impossible d'activer l'environnement conda '$targetCondaEnv': $($_.Exception.Message)"
        exit 1
    }
}
if (-not $condaActivated) {
    Write-Error "Environnement conda '$targetCondaEnv' non actif."
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = $null

# Detect project root (config + src) from current script or sibling numerai-project
$candidates = @(
    $ScriptDir,
    (Join-Path $ScriptDir "numerai-project")
)
foreach ($c in $candidates) {
    if ((Test-Path (Join-Path $c "config")) -and (Test-Path (Join-Path $c "src"))) {
        $RootDir = $c
        break
    }
}

if (-not $RootDir) {
    Write-Error "[ERROR] Cannot locate project root (config/src) from: $ScriptDir"
    exit 1
}

Set-Location $RootDir
Write-Host "Project root: $RootDir"

# Optional local credentials file (not tracked by git)
$keysCandidates = @(
    (Join-Path $ScriptDir "keys_local.ps1"),
    (Join-Path $RootDir "keys_local.ps1"),
    (Join-Path $RootDir "scripts/keys_local.ps1")
)
foreach ($candidate in $keysCandidates) {
    if (Test-Path $candidate) {
        Write-Host "Loading Numerai credentials from $candidate"
        . $candidate
        break
    }
}

# Resolve submission path (defaults to submission.csv, honor training.yaml if present)
$trainingCfgPath = Join-Path $RootDir "config/training.yaml"
$submissionPath = Join-Path $RootDir "submission.csv"
if ((Test-Path $trainingCfgPath) -and (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue)) {
    try {
        $trainingCfg = Get-Content -Raw -Path $trainingCfgPath | ConvertFrom-Yaml -ErrorAction Stop
        if ($trainingCfg.files.submission) {
            $submissionPath = $trainingCfg.files.submission
        }
    }
    catch {
        Write-Warning "Unable to read training.yaml: $($_.Exception.Message)"
    }
}
if (-not (Split-Path $submissionPath -IsAbsolute)) {
    $submissionPath = Join-Path $RootDir $submissionPath
}

if (-not (Test-Path $submissionPath)) {
    Write-Error "Submission file not found: $submissionPath"
    exit 1
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found in PATH."
    exit 1
}

$requiredEnv = @("NUMERAI_PUBLIC_ID", "NUMERAI_SECRET_KEY", "NUMERAI_MODEL_ID")
$missing = $requiredEnv | Where-Object { -not [System.Environment]::GetEnvironmentVariable($_) }
if ($missing) {
    Write-Error "Missing Numerai env vars: $($missing -join ', '). Export them before running."
    exit 1
}

Write-Host "Submitting file: $submissionPath"
$uploadScript = @"
import os
import sys
from numerapi import NumerAPI

pub = os.environ["NUMERAI_PUBLIC_ID"]
sec = os.environ["NUMERAI_SECRET_KEY"]
mid = os.environ["NUMERAI_MODEL_ID"]
spath = r"$($submissionPath)"

if not os.path.isfile(spath):
    sys.stderr.write(f"Missing submission file: {spath}\n")
    sys.exit(2)

napi = NumerAPI(pub, sec)
resp = napi.upload_predictions(spath, model_id=mid)
print("Upload:", resp)
"@

$uploadScript | python -
exit $LASTEXITCODE
