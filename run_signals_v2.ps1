$ErrorActionPreference = "Stop"

if (-not $env:NUMERAI_PROGRESS) { $env:NUMERAI_PROGRESS = "1" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = $null

# Optional conda activation (reuse same env name as run_me.ps1)
$targetCondaEnv = "lgbm-gpu"
# $targetCondaEnv = $env:NUMERAI_CONDA_ENV
# if (-not $targetCondaEnv) { $targetCondaEnv = "lgbm-gpu" }

if ($env:CONDA_DEFAULT_ENV -ne $targetCondaEnv) {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        try {
            function Resolve-EnvFile {
                param([string]$BaseDir)
                $candidates = @(
                    (Join-Path $BaseDir "numerai-project\environment.yml"),
                    (Join-Path $BaseDir "environment.yml")
                )
                foreach ($p in $candidates) {
                    if (Test-Path $p) { return $p }
                }
                return $null
            }
            function Ensure-CondaEnv {
                param([string]$EnvName, [string]$EnvFile)
                $envList = & $condaCmd "env" "list"
                $pattern = "^\s*$([regex]::Escape($EnvName))\s"
                $exists = $envList | Select-String -Pattern $pattern
                if (-not $exists) {
                    if (-not $EnvFile) {
                        Write-Error "Conda env '$EnvName' introuvable et aucun environment.yml trouvé pour le créer."
                        exit 1
                    }
                    Write-Host "Creating conda env '$EnvName' from $EnvFile"
                    & $condaCmd "env" "create" "-n" $EnvName "-f" $EnvFile
                    if ($LASTEXITCODE -ne 0) {
                        Write-Error "Echec creation conda env '$EnvName'."
                        exit 1
                    }
                }
            }
            $envFile = Resolve-EnvFile -BaseDir $ScriptDir
            Ensure-CondaEnv -EnvName $targetCondaEnv -EnvFile $envFile
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

$trainingCfgRel = "config/training_signals_v2.yaml"
$featuresCfgRel = "config/features_signals_v2.yaml"
$paramsRel = "config/program_input_params.yaml"

$trainingCfgPath = Join-Path $RootDir $trainingCfgRel
$featuresCfgPath = Join-Path $RootDir $featuresCfgRel
$paramsPath = Join-Path $RootDir $paramsRel

$trainPath = Join-Path $RootDir "data/signals_v2/train.parquet"
$livePath  = Join-Path $RootDir "data/signals_v2/live.parquet"

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
    $signalsVersion = $env:NUMERAI_SIGNALS_VERSION
    if (-not $signalsVersion) { $signalsVersion = "signals/v2.1" }
    $signalsVersion = $signalsVersion.TrimEnd("/")
    if ($signalsVersion -notmatch "/") { $signalsVersion = "signals/$signalsVersion" }

    Write-Host "Refreshing Numerai Signals datasets ($signalsVersion)..."
    $downloadScript = @"
from pathlib import Path
from numerapi import SignalsAPI

signals_version = r"$signalsVersion"
train_path = Path(r"$trainPath")
live_path = Path(r"$livePath")
train_path.parent.mkdir(parents=True, exist_ok=True)
live_path.parent.mkdir(parents=True, exist_ok=True)

api = SignalsAPI()

def download(src: str, dest: Path) -> None:
    try:
        api.download_dataset(src, str(dest))
        print(f"Downloaded {src} -> {dest}")
    except Exception as exc:
        print(f"DOWNLOAD_WARNING {src}: {exc}")

download(f"{signals_version}/train.parquet", train_path)
download(f"{signals_version}/live.parquet", live_path)
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
        --output "signals.pkl"
}
catch {
    Write-Error "Model upload pickle build failed: $($_.Exception.Message)"
    exit 1
}

$signalsNodeDir = Join-Path $RootDir "signals-node"
if (Test-Path $signalsNodeDir) {
    Copy-Item -Path (Join-Path $RootDir "signals.pkl") -Destination (Join-Path $signalsNodeDir "signals.pkl") -Force
    Write-Host "Copied signals.pkl to $signalsNodeDir"
}
else {
    Write-Warning "signals-node not found at $signalsNodeDir. Copy manually."
}
