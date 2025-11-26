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

# Check python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "python not found in PATH. Activate the proper conda/micromamba env (e.g. lgbm-gpu) and retry."
    exit 1
}

# Check numerai CLI
$numeraiCmd = Get-Command numerai -ErrorAction SilentlyContinue
if ($numeraiCmd) {
    Write-Host "numerai CLI detected."
} else {
    Write-Warning "numerai CLI not found; submission will rely on numerapi if needed."
}

###############################################################################
# GPU detection + VRAM
###############################################################################
if (-not $env:GPU_REQUIRED) { $env:GPU_REQUIRED = "1" }
$totalVramMb = 0
$freeVramMb  = 0

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "GPU detected:"
    & nvidia-smi
    $vramInfo = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
    $line = ($vramInfo -split "`n")[0].Trim()
    $parts = $line -split ","
    if ($parts.Length -ge 2) {
        $totalVramMb = [int]$parts[0].Trim()
        $freeVramMb  = [int]$parts[1].Trim()
        Write-Host "VRAM total: $totalVramMb MB, free: $freeVramMb MB"
    } else {
        Write-Warning "Unable to parse VRAM info; continuing with free VRAM = 0."
    }
} else {
    if ($env:GPU_REQUIRED -eq "1") {
        Write-Error "GPU_REQUIRED=1 but nvidia-smi not found. Install drivers or set GPU_REQUIRED=0 to force CPU."
        exit 1
    } else {
        Write-Warning "GPU not detected; continuing with GPU_REQUIRED=0."
    }
}

###############################################################################
# Download Numerai data if missing
###############################################################################
$trainPath = Join-Path $RootDir "data/numerai_training_data.parquet"
$tourPath  = Join-Path $RootDir "data/numerai_tournament_data.parquet"
if ((Test-Path $trainPath -PathType Leaf) -and (Test-Path $tourPath -PathType Leaf)) {
    Write-Host "Datasets already present, skipping download."
} else {
    Write-Host "Downloading Numerai datasets..."
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
    $tmpPy = Join-Path $env:TEMP ("numerai_download_" + [guid]::NewGuid().ToString() + ".py")
    Set-Content -Path $tmpPy -Value $pyScript -Encoding UTF8
    try {
        & python $tmpPy
    } finally {
        Remove-Item $tmpPy -ErrorAction SilentlyContinue
    }
}

###############################################################################
# Define pluq levels with VRAM requirements (heaviest -> lightest)
###############################################################################
$pluqLevels = @(
    @{ Name = "pluq4"; Params = "config/model_params_pluq4.yaml"; MinFreeVramMb = 7000 },
    @{ Name = "pluq3"; Params = "config/model_params_pluq3.yaml"; MinFreeVramMb = 5000 },
    @{ Name = "pluq2"; Params = "config/model_params_pluq2.yaml"; MinFreeVramMb = 3000 },
    @{ Name = "pluq1"; Params = "config/model_params_pluq1.yaml"; MinFreeVramMb = 1500 }
)

$eligibleLevels = if ($freeVramMb -gt 0) {
    $pluqLevels | Where-Object { $freeVramMb -ge $_.MinFreeVramMb }
} else {
    $pluqLevels
}

if (-not $eligibleLevels -or $eligibleLevels.Count -eq 0) {
    Write-Error "Not enough free VRAM for even pluq1 (needs >=1500 MB). Free: $freeVramMb MB."
    exit 1
}

###############################################################################
# Sweep eligible levels (heaviest to lightest)
###############################################################################
$best = $null
foreach ($lvl in $eligibleLevels) {
    $pfile = Join-Path $RootDir $lvl.Params
    if (-not (Test-Path $pfile -PathType Leaf)) {
        Write-Warning "Params file missing for $($lvl.Name): $pfile. Skipping."
        continue
    }
    Write-Host "Testing level $($lvl.Name) (requires >= $($lvl.MinFreeVramMb) MB free) with params $($lvl.Params)..."
    try {
        & python "src/train.py" --config "config/training.yaml" --params $lvl.Params --features "config/features.yaml"
    } catch {
        Write-Warning "Training failed for $($lvl.Name): $($_.Exception.Message)"
        continue
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Training exited with code $LASTEXITCODE for $($lvl.Name). Trying next level."
        continue
    }
    Write-Host "Level $($lvl.Name) succeeded."
    $best = $lvl
    break
}

if (-not $best) {
    Write-Error "All eligible pluq levels failed. No model trained."
    exit 1
}

###############################################################################
# Prediction with best level
###############################################################################
try {
    & python "src/predict.py" --config "config/training.yaml" --params $best.Params --features "config/features.yaml"
} catch {
    Write-Error "Prediction failed with $($best.Name): $($_.Exception.Message)"
    exit 1
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Prediction exited with code $LASTEXITCODE using $($best.Name)."
    exit 1
}

###############################################################################
# Optional Docker block
###############################################################################
if ($env:DOCKER -eq "1") {
    Write-Host "DOCKER=1 detected. Building docker image..."
    docker build -t numerai:latest docker
    Write-Host "Running docker container..."
    docker run --rm -v "$RootDir/data:/app/data" -v "$RootDir/models:/app/models" numerai:latest
}

Write-Host "=== Summary ==="
Write-Host "Selected level : $($best.Name)"
Write-Host "Params file    : $($best.Params)"
Write-Host "VRAM total/free: $totalVramMb MB / $freeVramMb MB"
Write-Host "submission.csv generated (see config/training.yaml for path)"
