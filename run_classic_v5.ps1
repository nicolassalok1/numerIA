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

$trainingCfgRel = "config/training.yaml"
$featuresCfgRel = "config/features.yaml"
$paramsRel = "config/program_input_params.yaml"

$trainingCfgPath = Join-Path $RootDir $trainingCfgRel
$featuresCfgPath = Join-Path $RootDir $featuresCfgRel
$paramsPath = Join-Path $RootDir $paramsRel

$trainPath = Join-Path $RootDir "data/numerai_training_data.parquet"
$livePath  = Join-Path $RootDir "data/numerai_tournament_data.parquet"

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
    $classicVersion = $env:NUMERAI_CLASSIC_VERSION
    if (-not $classicVersion) { $classicVersion = "v5.2" }
    $classicVersion = $classicVersion.TrimEnd("/")

    Write-Host "Refreshing Numerai Classic datasets ($classicVersion)..."
    $downloadScript = @"
from pathlib import Path
from numerapi import NumerAPI

classic_version = r"$classicVersion"
train_path = Path(r"$trainPath")
live_path = Path(r"$livePath")
train_path.parent.mkdir(parents=True, exist_ok=True)
live_path.parent.mkdir(parents=True, exist_ok=True)

api = NumerAPI()

def download(src: str, dest: Path) -> None:
    try:
        api.download_dataset(src, str(dest))
        print(f"Downloaded {src} -> {dest}")
    except Exception as exc:
        print(f"DOWNLOAD_WARNING {src}: {exc}")

download(f"{classic_version}/train.parquet", train_path)
download(f"{classic_version}/live.parquet", live_path)
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
        --output "salok1_classic.pkl"
}
catch {
    Write-Error "Model upload pickle build failed: $($_.Exception.Message)"
    exit 1
}

$classicNodeDir = Join-Path $RootDir "classic-node"
if (Test-Path $classicNodeDir) {
    Copy-Item -Path (Join-Path $RootDir "salok1_classic.pkl") -Destination (Join-Path $classicNodeDir "salok1_classic.pkl") -Force
    Write-Host "Copied salok1_classic.pkl to $classicNodeDir"
}
else {
    Write-Warning "classic-node not found at $classicNodeDir. Copy manually."
}

$tgrv2NodeDir = Join-Path $RootDir "tgrv2-node"
if (Test-Path $tgrv2NodeDir) {
    Copy-Item -Path (Join-Path $RootDir "salok1_classic.pkl") -Destination (Join-Path $tgrv2NodeDir "tgrv2.pkl") -Force
    Write-Host "Copied salok1_classic.pkl to $tgrv2NodeDir as tgrv2.pkl"
}
else {
    Write-Warning "tgrv2-node not found at $tgrv2NodeDir. Copy manually."
}
