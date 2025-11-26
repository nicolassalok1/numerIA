param(
    [string]$ParamsFile = ""
)

$ErrorActionPreference = "Stop"

$env:NUMERAI_PUBLIC_ID="RCB3JR7KWN2BS2EOYAEHIHXBPAWCDYLS"
$env:NUMERAI_SECRET_KEY="LJ7IIKM4VF7R7RBTB2EMM2X2VTKNDGC4HXHZVGLMVW3W5IIGDFDZNIJOWO6BQCY3"
$env:NUMERAI_MODEL_ID="ecb01105-5985-43bb-b76e-445d94c22928"

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

# Resolve data paths
$trainingCfgRel    = "config/training.yaml"
$featuresCfgRel    = "config/features.yaml"

$trainingCfgPath   = Join-Path $RootDir $trainingCfgRel
$trainPath         = Join-Path $RootDir "data/numerai_training_data.parquet"
$tourPath          = Join-Path $RootDir "data/numerai_tournament_data.parquet"
$submissionPath    = Join-Path $RootDir "submission.csv"

if (Test-Path $trainingCfgPath) {
    try {
        $trainingCfg = Get-Content -Raw -Path $trainingCfgPath | ConvertFrom-Yaml
        if ($trainingCfg.files.train)       { $trainPath = $trainingCfg.files.train }
        if ($trainingCfg.files.tournament)  { $tourPath = $trainingCfg.files.tournament }
        if ($trainingCfg.files.submission)  { $submissionPath = $trainingCfg.files.submission }
    }
    catch { Write-Warning "Unable to parse training.yaml" }
}

if (-not (Split-Path $trainPath -IsAbsolute))      { $trainPath      = Join-Path $RootDir $trainPath }
if (-not (Split-Path $tourPath -IsAbsolute))       { $tourPath       = Join-Path $RootDir $tourPath }
if (-not (Split-Path $submissionPath -IsAbsolute)) { $submissionPath = Join-Path $RootDir $submissionPath }

# Check tooling
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found in PATH."
    exit 1
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
# 2) Hardcore VRAM Tier
###############################################################################
# RULES:
# - If free VRAM >= 7200 MB → use pluq4_hardcore
# - Else → fallback to model_params_pluq4060.yaml

if ($ParamsFile) {
    $tier = "CUSTOM"
    $paramsPath = $ParamsFile
}
elseif ($freeVramMb -ge 7200) {
    $tier = "PLUQ4-HARDCORE"
    $paramsPath = "config/model_params_pluq4_hardcore.yaml"
}
else {
    $tier = "PLUQ4060"
    $paramsPath = "config/model_params_pluq4060.yaml"
}

if (-not (Split-Path $paramsPath -IsAbsolute)) {
    $paramsPath = Join-Path $RootDir $paramsPath
}

Write-Host "Selected Tier: $tier"
Write-Host "Params file: $paramsPath"
Write-Host "Training config: $trainingCfgPath"
Write-Host "Features config: $(Join-Path $RootDir $featuresCfgRel)"

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
Write-Host "  Tier: $tier"
Write-Host "  Free VRAM: $freeVramMb MB"
Write-Host "  Params used: $paramsPath"
