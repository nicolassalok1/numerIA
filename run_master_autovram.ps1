$ErrorActionPreference = "Stop"

# Locate project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Test-Path (Join-Path $ScriptDir "config")) -and (Test-Path (Join-Path $ScriptDir "src"))) {
    $RootDir = $ScriptDir
}
elseif (Test-Path (Join-Path $ScriptDir "numerai-project\config")) {
    $RootDir = Join-Path $ScriptDir "numerai-project"
}
else {
    Write-Error "[ERROR] Cannot locate project root (config/ and src/ expected)."
    exit 1
}
Set-Location $RootDir
Write-Host "Project root: $RootDir"

# Tooling checks
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "python not found in PATH. Please ensure your conda/env is activated."
    exit 1
}

$numeraiCmd = Get-Command numerai -ErrorAction SilentlyContinue
$HAS_NUMERAI = if ($numeraiCmd) { 1 } else { 0 }

###############################################################################
# 1) GPU detection + VRAM query
###############################################################################
if (-not $env:GPU_REQUIRED) {
    $env:GPU_REQUIRED = "1"
}

$freeVramMb = 0
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "GPU detected:"
    & nvidia-smi

    $vramInfo = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
    $vramLine = ($vramInfo -split "`n")[0].Trim()
    $parts = $vramLine -split ","
    if ($parts.Length -ge 2) {
        $totalVramMb = [int]$parts[0].Trim()
        $freeVramMb = [int]$parts[1].Trim()
        Write-Host "VRAM total: $totalVramMb MB, free: $freeVramMb MB"
    }
    else {
        Write-Warning "Unable to parse VRAM info, defaulting to 0 MB free."
        $freeVramMb = 0
    }
}
else {
    if ($env:GPU_REQUIRED -eq "1") {
        Write-Error "GPU required but nvidia-smi not found. Install NVIDIA drivers or set GPU_REQUIRED=0 to force CPU."
        exit 1
    }
    else {
        Write-Warning "GPU not detected; continuing (may fall back to CPU)."
    }
}

###############################################################################
# 2) Auto-VRAM based LightGBM model configuration
###############################################################################
$tier = "safe"
if ($freeVramMb -ge 7000) {
    $tier = "aggressive"
}
elseif ($freeVramMb -ge 5000) {
    $tier = "high"
}
elseif ($freeVramMb -ge 3000) {
    $tier = "medium"
}

switch ($tier) {
    "aggressive" {
        $tierParams = @{
            num_leaves        = 512
            max_bin           = 511
            learning_rate     = 0.02
            n_estimators      = 2500
            feature_fraction  = 0.9
            bagging_fraction  = 0.9
            bagging_freq      = 1
            min_data_in_leaf  = 60
        }
    }
    "high" {
        $tierParams = @{
            num_leaves        = 256
            max_bin           = 511
            learning_rate     = 0.03
            n_estimators      = 2000
            feature_fraction  = 0.8
            bagging_fraction  = 0.8
            bagging_freq      = 1
            min_data_in_leaf  = 100
        }
    }
    "medium" {
        $tierParams = @{
            num_leaves        = 128
            max_bin           = 255
            learning_rate     = 0.05
            n_estimators      = 1500
            feature_fraction  = 0.7
            bagging_fraction  = 0.7
            bagging_freq      = 1
            min_data_in_leaf  = 150
        }
    }
    default {
        $tierParams = @{
            num_leaves        = 64
            max_bin           = 255
            learning_rate     = 0.05
            n_estimators      = 1000
            feature_fraction  = 0.6
            bagging_fraction  = 0.6
            bagging_freq      = 1
            min_data_in_leaf  = 200
        }
    }
}

$yamlLines = @()
$yamlLines += "model:"
$yamlLines += "  boosting_type: gbdt"
$yamlLines += "  objective: regression"
$yamlLines += "  metric: mae"
$yamlLines += "  device_type: gpu"
$yamlLines += "  gpu_platform_id: 0"
$yamlLines += "  gpu_device_id: 0"
$yamlLines += "  max_depth: -1"
$yamlLines += "  min_sum_hessian_in_leaf: 0.001"
$yamlLines += "  lambda_l1: 0.0"
$yamlLines += "  lambda_l2: 5.0"
$yamlLines += "  min_gain_to_split: 0.0"
$yamlLines += "  verbosity: -1"
foreach ($kv in $tierParams.GetEnumerator()) {
    $yamlLines += "  $($kv.Key): $($kv.Value)"
}
$yamlContent = ($yamlLines -join "`n")
$paramsPath = "config/model_params.yaml"
Set-Content -Path $paramsPath -Value $yamlContent -Encoding UTF8

Write-Host "Selected auto-VRAM tier: $tier (free VRAM: $freeVramMb MB)"
Write-Host "Using params file: $paramsPath"

###############################################################################
# 3) Numerai data download (if missing)
###############################################################################
$trainPath = Join-Path $RootDir "data/numerai_training_data.parquet"
$tourPath = Join-Path $RootDir "data/numerai_tournament_data.parquet"
if ((Test-Path $trainPath -PathType Leaf) -and (Test-Path $tourPath -PathType Leaf)) {
    Write-Host "Datasets already present, skipping download."
}
else {
    $pyScript = @"
from pathlib import Path
import numerapi

data_dir = Path("data")
train_dst = data_dir / "numerai_training_data.parquet"
tour_dst = data_dir / "numerai_tournament_data.parquet"
data_dir.mkdir(parents=True, exist_ok=True)

napi = numerapi.NumerAPI()
datasets = napi.list_datasets()

def pick_dataset(candidates, exclude=None):
    exclude = exclude or []
    for suffix in candidates:
        matches = sorted(
            d for d in datasets
            if d.endswith(suffix)
            and not any(ex.lower() in d.lower() for ex in exclude)
        )
        if matches:
            return matches[-1]
    raise SystemExit(f"No dataset found for {candidates}. Examples: {datasets[:5]}")

train_ds = pick_dataset(
    ["train.parquet", "numerai_training_data.parquet", "latest_numerai_training_data.parquet"],
    exclude=["benchmark", "example", "pred"]
)
tour_ds = pick_dataset(
    ["live.parquet", "numerai_tournament_data.parquet", "latest_numerai_tournament_data.parquet"],
    exclude=["benchmark", "example", "pred"]
)

print(f"Downloading train: {train_ds}")
napi.download_dataset(train_ds, str(train_dst))
print(f"Downloading tournament: {tour_ds}")
napi.download_dataset(tour_ds, str(tour_dst))
print("Downloads complete.")
"@
    $tmpPy = Join-Path $env:TEMP "numerai_download_$([System.Guid]::NewGuid().ToString()).py"
    Set-Content -Path $tmpPy -Value $pyScript -Encoding UTF8
    try {
        & python $tmpPy
    }
    finally {
        Remove-Item $tmpPy -ErrorAction SilentlyContinue
    }
}

###############################################################################
# 4) Training and prediction
###############################################################################
try {
    & python "src/train.py" --config "config/training.yaml" --params $paramsPath --features "config/features.yaml"
}
catch {
    Write-Error "Training failed: $($_.Exception.Message)"
    exit 1
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training exited with code $LASTEXITCODE"
    exit 1
}

try {
    & python "src/predict.py" --config "config/training.yaml" --params $paramsPath --features "config/features.yaml"
}
catch {
    Write-Error "Prediction failed: $($_.Exception.Message)"
    exit 1
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Prediction exited with code $LASTEXITCODE"
    exit 1
}

###############################################################################
# 5) Optional numerai CLI submission
###############################################################################
if ($HAS_NUMERAI -eq 1 -and $env:NUMERAI_PUBLIC_ID -and $env:NUMERAI_SECRET_KEY -and $env:NUMERAI_MODEL_ID) {
    try {
        numerai submit --model-id $env:NUMERAI_MODEL_ID --public-id $env:NUMERAI_PUBLIC_ID --secret-key $env:NUMERAI_SECRET_KEY "submission.csv"
    }
    catch {
        Write-Warning "Numerai CLI submission failed: $($_.Exception.Message)"
    }
}
else {
    Write-Host "Local submission skipped (numerai CLI or secrets missing)."
}

Write-Host "Summary:"
Write-Host "  Tier: $tier"
Write-Host "  Free VRAM used: $freeVramMb MB"
Write-Host "  Params file: $paramsPath"
Write-Host "  Submission: submission.csv (see config/training.yaml)"
