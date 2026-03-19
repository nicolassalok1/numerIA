$ErrorActionPreference = "Stop"

# ============================================================
# 1. ACTIVER L'ENVIRONNEMENT CONDA
# ============================================================
$targetCondaEnv = "numerai-env"

if ($env:CONDA_DEFAULT_ENV -ne $targetCondaEnv) {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) {
        Write-Error "conda introuvable dans PATH. Active l'environnement '$targetCondaEnv' manuellement avant de lancer."
        exit 1
    }
    (& $condaCmd "shell.powershell" "hook") | Out-String | Invoke-Expression
    conda activate $targetCondaEnv
    Write-Host "Conda env active: $targetCondaEnv"
} else {
    Write-Host "Conda env deja active: $targetCondaEnv"
}

$RootDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "numerai-project"
if (-not (Test-Path (Join-Path $RootDir "src"))) {
    $RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $RootDir
Write-Host "Project root: $RootDir"

# ============================================================
# 2. INSTALLER LES NOUVELLES DEPENDANCES (si manquantes)
# ============================================================
Write-Host ""
Write-Host "Verification des dependances..."
$missingPkgs = @()
$checkScript = "
import importlib, sys
for pkg in ['xgboost', 'optuna']:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(pkg)
"
$missing = python -c $checkScript 2>$null
if ($missing) {
    Write-Host "Installation des packages manquants: $missing"
    pip install $missing
} else {
    Write-Host "Toutes les dependances sont installees."
}

# ============================================================
# 3. TELECHARGER LES DONNEES NUMERAI (si pas deja fait)
# ============================================================
$trainPath = Join-Path $RootDir "data/numerai_training_data.parquet"
$livePath  = Join-Path $RootDir "data/numerai_tournament_data.parquet"

if ((-not (Test-Path $trainPath)) -or (-not (Test-Path $livePath))) {
    Write-Host ""
    Write-Host "Telechargement des donnees Numerai..."
    $dlScript = @"
from pathlib import Path
from numerapi import NumerAPI
napi = NumerAPI()
Path(r"$RootDir/data").mkdir(exist_ok=True)
print("Downloading training data...")
napi.download_dataset("v5.0/train.parquet", r"$trainPath")
print("Downloading live data...")
napi.download_dataset("v5.0/live.parquet", r"$livePath")
print("Done.")
"@
    $dlScript | python -
} else {
    Write-Host "Donnees deja presentes, skip download. (Supprime data/ pour forcer le refresh)"
}

# ============================================================
# 4. (OPTIONNEL) OPTIMISER LES HYPERPARAMS AVEC OPTUNA
#    Decommente les 2 lignes ci-dessous pour activer.
#    ~30-60 min selon ta GPU.
# ============================================================
# Write-Host ""
# Write-Host "=== OPTUNA OPTIMIZATION ==="
# python src/optimize.py --n-trials 30
# $paramsFile = "config/optimized_params.yaml"

# Si Optuna n'est pas lance, utiliser les params par defaut (deja ameliores)
if (-not $paramsFile) {
    $paramsFile = "config/program_input_params.yaml"
}

# ============================================================
# 5. ENTRAINER LES MODELES
# ============================================================
Write-Host ""
Write-Host "=== TRAINING ==="
Write-Host "Params: $paramsFile"
python src/train.py `
    --config config/training.yaml `
    --params $paramsFile `
    --features config/features.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training echoue (exit code $LASTEXITCODE)"
    exit 1
}

# ============================================================
# 6. GENERER LES PREDICTIONS
# ============================================================
Write-Host ""
Write-Host "=== PREDICTION ==="
python src/predict.py `
    --config config/training.yaml `
    --params $paramsFile `
    --features config/features.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Error "Prediction echouee (exit code $LASTEXITCODE)"
    exit 1
}

# ============================================================
# 7. DIAGNOSTICS
# ============================================================
Write-Host ""
Write-Host "=== DIAGNOSTICS ==="
python src/diagnostics.py `
    --config config/training.yaml `
    --params $paramsFile `
    --features config/features.yaml

# ============================================================
# 8. SOUMETTRE A NUMERAI
#    Necessite les variables d'environnement:
#      NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY, NUMERAI_MODEL_ID
#    Tu peux les definir dans keys_local.ps1 ou les exporter avant.
# ============================================================
$keysFile = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "keys_local.ps1"
if (Test-Path $keysFile) {
    Write-Host ""
    Write-Host "Chargement des credentials depuis $keysFile"
    . $keysFile
}

$submissionPath = Join-Path $RootDir "submission.csv"

if ($env:NUMERAI_PUBLIC_ID -and $env:NUMERAI_SECRET_KEY -and $env:NUMERAI_MODEL_ID) {
    if (Test-Path $submissionPath) {
        Write-Host ""
        Write-Host "=== UPLOAD SUBMISSION ==="
        $uploadScript = @"
import os
from numerapi import NumerAPI
napi = NumerAPI(os.environ['NUMERAI_PUBLIC_ID'], os.environ['NUMERAI_SECRET_KEY'])
resp = napi.upload_predictions(r"$submissionPath", model_id=os.environ['NUMERAI_MODEL_ID'])
print("Upload:", resp)
"@
        $uploadScript | python -
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Upload echoue"
            exit 1
        }
    } else {
        Write-Error "submission.csv introuvable: $submissionPath"
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "SKIP upload: credentials Numerai non definies."
    Write-Host "  Definis NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY, NUMERAI_MODEL_ID"
    Write-Host "  ou cree un fichier keys_local.ps1 a la racine du repo."
}

# ============================================================
# RESUME
# ============================================================
Write-Host ""
Write-Host "========================================"
Write-Host "  PIPELINE TERMINE"
Write-Host "========================================"
Write-Host "  Params utilises: $paramsFile"
Write-Host "  Submission:      $submissionPath"
Write-Host ""
Write-Host "  Commandes utiles:"
Write-Host "    python src/optimize.py --n-trials 50   # Optimiser les hyperparams"
Write-Host "    python src/diagnostics.py              # Re-lancer les diagnostics"
Write-Host "========================================"
