$ErrorActionPreference = "Stop"

if (-not $env:NUMERAI_PROGRESS) { $env:NUMERAI_PROGRESS = "1" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = $null

# Optional conda activation (reuse same env name as run_me.ps1)
$targetCondaEnv = $env:NUMERAI_CONDA_ENV
if (-not $targetCondaEnv) { $targetCondaEnv = "lgbm-gpu" }

if ($env:CONDA_DEFAULT_ENV -ne $targetCondaEnv) {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        try {
            (& $condaCmd "shell.powershell" "hook") | Out-String | Invoke-Expression
            conda activate $targetCondaEnv | Out-Null
            Write-Host "Activated conda env: $targetCondaEnv"
        }
        catch {
            Write-Warning "Conda activation failed: $($_.Exception.Message)"
        }
    }
}

# Locate project root (numerai-project)
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
    Write-Error "[ERROR] Cannot locate project root (config/src) from script directory: $ScriptDir"
    exit 1
}

Set-Location $RootDir
Write-Host "Project root: $RootDir"

$trainingCfgRel = "config/training_crypto_v2.yaml"
$featuresCfgRel = "config/features_crypto_v2.yaml"
$paramsRel = "config/program_input_params.yaml"

$trainingCfgPath = Join-Path $RootDir $trainingCfgRel
$featuresCfgPath = Join-Path $RootDir $featuresCfgRel
$paramsPath = Join-Path $RootDir $paramsRel

$trainPath = Join-Path $RootDir "data/crypto_v2/r1198__crypto_v2_0_train.parquet"
$livePath  = Join-Path $RootDir "data/crypto_v2/r1198__crypto_v2_0_live.parquet"

if (Test-Path $trainingCfgPath) {
    if (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue) {
        $trainingCfg = Get-Content -Raw -Path $trainingCfgPath | ConvertFrom-Yaml -ErrorAction SilentlyContinue
        if ($trainingCfg) {
            if ($trainingCfg.files.train)      { $trainPath = $trainingCfg.files.train }
            if ($trainingCfg.files.tournament) { $livePath  = $trainingCfg.files.tournament }
        }
    }
}
if (-not (Split-Path $trainPath -IsAbsolute)) { $trainPath = Join-Path $RootDir $trainPath }
if (-not (Split-Path $livePath -IsAbsolute))  { $livePath  = Join-Path $RootDir $livePath }

if (-not $env:SKIP_DATA_REFRESH) {
    Write-Host "Refreshing Numerai Crypto v2 datasets..."
    $downloadScript = @"
from pathlib import Path
from numerapi import CryptoAPI

train_path = Path(r"$trainPath")
live_path = Path(r"$livePath")
train_path.parent.mkdir(parents=True, exist_ok=True)
live_path.parent.mkdir(parents=True, exist_ok=True)

api = CryptoAPI()

def download(src: str, dest: Path) -> None:
    try:
        api.download_dataset(src, str(dest))
        print(f"Downloaded {src} -> {dest}")
    except Exception as exc:
        print(f"DOWNLOAD_WARNING {src}: {exc}")

download("v2.0/train.parquet", train_path)
download("v2.0/live.parquet", live_path)
"@
    $downloadScript | python -
}

Write-Host "Training config: $trainingCfgPath"
Write-Host "Features config: $featuresCfgPath"
Write-Host "Params file: $paramsPath"

try {
    & python "src/train.py" `
        --config $trainingCfgPath `
        --params $paramsPath `
        --features $featuresCfgPath
}
catch {
    Write-Error "Training failed: $($_.Exception.Message)"
    exit 1
}

try {
    & python "src/model_upload.py" `
        --models-dir "models" `
        --training-config $trainingCfgPath `
        --output "salok1.pkl"
}
catch {
    Write-Error "Model upload pickle build failed: $($_.Exception.Message)"
    exit 1
}

$cryptoNodeDir = Join-Path $RootDir "crypto-node"
if (Test-Path $cryptoNodeDir) {
    Copy-Item -Path (Join-Path $RootDir "salok1.pkl") -Destination (Join-Path $cryptoNodeDir "salok1.pkl") -Force
    Write-Host "Copied salok1.pkl to $cryptoNodeDir"
}
else {
    Write-Warning "crypto-node not found at $cryptoNodeDir. Copy manually."
}
