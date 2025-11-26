$ErrorActionPreference = "Stop"

# Locate project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ScriptDir
if ((Test-Path (Join-Path $ScriptDir "config")) -and (Test-Path (Join-Path $ScriptDir "src"))) {
    $RootDir = $ScriptDir
}
elseif ((Test-Path (Join-Path $ParentDir "config")) -and (Test-Path (Join-Path $ParentDir "src"))) {
    $RootDir = $ParentDir
}
elseif (Test-Path (Join-Path $ScriptDir "numerai-project\config")) {
    $RootDir = Join-Path $ScriptDir "numerai-project"
}
else {
    Write-Error "[ERROR] Cannot locate project root."
    exit 1
}
Set-Location $RootDir

Write-Host "Project root: $RootDir"

# Optional local credentials file (not tracked by git)
$localKeysPath = Join-Path $ScriptDir "keys_local.ps1"
if (Test-Path $localKeysPath) {
    Write-Host "Loading Numerai credentials from $localKeysPath"
    . $localKeysPath
}

# Resolve data paths
$trainingCfgRel    = "config/training.yaml"
$featuresCfgRel    = "config/features.yaml"

$trainingCfgPath   = Join-Path $RootDir $trainingCfgRel
$trainPath         = Join-Path $RootDir "data/numerai_training_data.parquet"
$tourPath          = Join-Path $RootDir "data/numerai_tournament_data.parquet"
$submissionPath    = Join-Path $RootDir "submission.csv"

if (Test-Path $trainingCfgPath) {
    if (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue) {
        $trainingCfg = Get-Content -Raw -Path $trainingCfgPath | ConvertFrom-Yaml -ErrorAction SilentlyContinue
        if ($trainingCfg) {
            if ($trainingCfg.files.train)       { $trainPath = $trainingCfg.files.train }
            if ($trainingCfg.files.tournament)  { $tourPath = $trainingCfg.files.tournament }
            if ($trainingCfg.files.submission)  { $submissionPath = $trainingCfg.files.submission }
        }
    }
}

if (-not (Split-Path $trainPath -IsAbsolute))      { $trainPath      = Join-Path $RootDir $trainPath }
if (-not (Split-Path $tourPath -IsAbsolute))       { $tourPath       = Join-Path $RootDir $tourPath }
if (-not (Split-Path $submissionPath -IsAbsolute)) { $submissionPath = Join-Path $RootDir $submissionPath }

# Refresh Numerai datasets unless explicitly skipped
if (-not $env:SKIP_DATA_REFRESH) {
    Write-Host "Refreshing Numerai datasets (train & live)..."
    $downloadScript = @"
import os
from pathlib import Path
from numerapi import NumerAPI

train_path = Path(r"$trainPath")
live_path = Path(r"$tourPath")
train_path.parent.mkdir(parents=True, exist_ok=True)
live_path.parent.mkdir(parents=True, exist_ok=True)

napi = NumerAPI()

def download(src: str, dest: Path) -> None:
    try:
        napi.download_dataset(src, str(dest))
        print(f"Downloaded {src} -> {dest}")
    except Exception as exc:
        print(f"DOWNLOAD_WARNING {src}: {exc}")

download("v5.0/train.parquet", train_path)
download("v5.0/live.parquet", live_path)
"@
    $downloadScript | python -
}

# Check tooling
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found in PATH."
    exit 1
}

# Check credentials presence before doing any heavy work
$requiredEnv = @("NUMERAI_PUBLIC_ID", "NUMERAI_SECRET_KEY", "NUMERAI_MODEL_ID")
$missing = $requiredEnv | Where-Object { -not [System.Environment]::GetEnvironmentVariable($_) }
if ($missing) {
    Write-Error "Missing Numerai env vars: $($missing -join ', '). Export them before running."
    exit 1
}

# Quick auth sanity check to fail fast on expired/invalid keys
$authScript = @'
import os, sys
import logging
from numerapi import NumerAPI

pub = os.environ["NUMERAI_PUBLIC_ID"]
sec = os.environ["NUMERAI_SECRET_KEY"]

# Silence numerapi noisy logging for limited-scope keys
logging.getLogger("numerapi").setLevel(logging.CRITICAL)

def is_invalid_session(err: Exception) -> bool:
    msg = str(err).lower()
    return "invalid" in msg or "expired" in msg or "forbidden" in msg

try:
    napi = NumerAPI(pub, sec)
    models_resp = napi.get_models() or []
    model_ids = []
    for m in models_resp:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("model_id")
            if mid:
                model_ids.append(str(mid))
        else:
            model_ids.append(str(m))
    print("Auth OK (models list) Models:", ", ".join(model_ids) or "<none>")
except Exception as exc:
    msg = str(exc)
    if is_invalid_session(exc):
        print("AUTH_ERROR_INVALID:", exc)
        sys.exit(3)
    if "Insufficient permission for read_user_info" in msg:
        # Key scoped for upload only; auth is fine for uploads.
        print("Auth OK (limited scope: upload only).")
        sys.exit(0)
    print("AUTH_WARNING_CONTINUE:", exc)
    sys.exit(0)
'@
$authScript | python -
if ($LASTEXITCODE -ne 0) {
    Write-Error "Numerai credentials invalid/expirées. Regénère une Secret Key et vérifie le Model ID."
    exit $LASTEXITCODE
}

function Show-YamlConfig {
    param(
        [string]$Label,
        [string]$Path
    )
    Write-Host ""
    Write-Host "# --- $Label ---"
    Write-Host "# Path: $Path"
    if (-not (Test-Path $Path)) {
        Write-Host "# [ABSENT] fichier introuvable"
        return
    }
    Get-Content -Path $Path | ForEach-Object { "# $_" }
}

###############################################################################
# 1) GPU detection + VRAM Query
###############################################################################
$freeVramMb = 0

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "GPU detected:"
    & nvidia-smi

    $vramInfo = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
    $parts = ($vramInfo -split "`n")[0].Split(",")

    if ($parts.Length -ge 2) {
        $totalVramMb = [int]$parts[0].Trim()
        $freeVramMb  = [int]$parts[1].Trim()
        Write-Host "VRAM total: $totalVramMb MB, free: $freeVramMb MB"
    }
}

###############################################################################
# 2) Model parameters (YAML source of truth, fixe par défaut)
###############################################################################
$paramsPath = Join-Path $RootDir "config/program_input_params.yaml"
Write-Host "Params file: $paramsPath"
Write-Host "Training config: $trainingCfgPath"
Write-Host "Features config: $(Join-Path $RootDir $featuresCfgRel)"

# Affiche toute la config utilisée (style commentaires YAML)
Write-Host ""
Write-Host "# =================================================================="
Write-Host "# CONFIG EFFECTIVE DU RUN"
Write-Host "# =================================================================="
Write-Host "# Fichiers utilisés:"
Write-Host "# - training.yaml  : $trainingCfgPath"
Write-Host "# - features.yaml  : $(Join-Path $RootDir $featuresCfgRel)"
Write-Host "# - params (LGBM)  : $paramsPath"
Write-Host "# - data train     : $trainPath"
Write-Host "# - data tournament: $tourPath"
Write-Host "# - submission     : $submissionPath"
Write-Host "# GPU free VRAM    : $freeVramMb MB"
Write-Host "# Params LGBM      : $paramsPath"
Show-YamlConfig -Label "training.yaml (config globale)" -Path $trainingCfgPath
Show-YamlConfig -Label "features.yaml (features utilisées)" -Path (Join-Path $RootDir $featuresCfgRel)
Show-YamlConfig -Label "params LightGBM (hyperparamètres)" -Path $paramsPath

###############################################################################
# 3) Training + Prediction
###############################################################################
try {
    & python "src/train.py" `
        --config $trainingCfgPath `
        --params $paramsPath `
        --features (Join-Path $RootDir $featuresCfgRel)
}
catch {
    Write-Error "Training failed: $($_.Exception.Message)"
    exit 1
}

try {
    & python "src/predict.py" `
        --config $trainingCfgPath `
        --params $paramsPath `
        --features (Join-Path $RootDir $featuresCfgRel)
}
catch {
    Write-Error "Prediction failed: $($_.Exception.Message)"
    exit 1
}

###############################################################################
# 4) Submission
###############################################################################
@"
import os
from numerapi import NumerAPI

pub   = os.environ["NUMERAI_PUBLIC_ID"]
sec   = os.environ["NUMERAI_SECRET_KEY"]
mid   = os.environ["NUMERAI_MODEL_ID"]
spath = r"$($submissionPath)"

napi = NumerAPI(pub, sec)
resp = napi.upload_predictions(spath, model_id=mid)
print("Upload:", resp)
"@ | python -

###############################################################################
Write-Host "Summary:"
Write-Host "  Free VRAM: $freeVramMb MB"
Write-Host "  Params used: $paramsPath"
