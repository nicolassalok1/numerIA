$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $RootDir "numerai-project"
$ScriptPath = Join-Path $ProjectDir "src\model_upload.py"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$PythonCmd = $null
$PythonArgs = @()
$CondaEnvName = if ($env:MODEL_UPLOAD_CONDA_ENV) { $env:MODEL_UPLOAD_CONDA_ENV } else { "numerai-env" }
$EnvFile = Join-Path $ProjectDir "environment.yml"
$UseConda = $false
$CondaEnvsRoot = Join-Path $RootDir ".conda_envs"
$CondaEnvPath = Join-Path $CondaEnvsRoot $CondaEnvName
$ForceCondaUpdate = $env:MODEL_UPLOAD_CONDA_UPDATE -eq "1"
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCmd) {
  if (-not (Test-Path $CondaEnvsRoot)) {
    New-Item -ItemType Directory -Path $CondaEnvsRoot | Out-Null
  }
  $env:CONDA_NO_PLUGINS = "1"
  $env:CONDA_NOTICES = "0"
  $env:CONDA_SOLVER = "classic"
  $env:CONDA_REPORT_ERRORS = "false"
  $env:CONDA_ALWAYS_YES = "true"
  $env:CONDA_OFFLINE = "1"
  $env:CONDA_ENVS_PATH = $CondaEnvsRoot
  $CondaPkgsRoot = Join-Path $RootDir ".conda_pkgs"
  if (-not (Test-Path $CondaPkgsRoot)) {
    New-Item -ItemType Directory -Path $CondaPkgsRoot | Out-Null
  }
  $env:CONDA_PKGS_DIRS = $CondaPkgsRoot
  if ($ForceCondaUpdate -and (Test-Path $EnvFile)) {
    Write-Host "Updating conda env '$CondaEnvName' from $EnvFile"
    conda --no-plugins env update -p $CondaEnvPath -f $EnvFile --solver classic
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "Conda env update failed for $CondaEnvName. Continuing with existing environment."
    }
  }
  $KnownCondaPaths = @(
    (Join-Path $CondaEnvPath "python.exe"),
    "D:\Programmes\Miniconda3\envs\$CondaEnvName\python.exe",
    "C:\Users\nicol\.conda\envs\$CondaEnvName\python.exe"
  ).Where({ $_ -and (Test-Path $_) })

  if ($KnownCondaPaths.Count -gt 0) {
    $PythonCmd = $KnownCondaPaths[0]
    $PythonArgs = @()
    $UseConda = $true
  }
}

if (-not $UseConda -and (Test-Path $VenvPython)) {
  $PythonCmd = $VenvPython
  $PythonArgs = @()
} elseif (-not $UseConda) {
  $OverridePython = $env:MODEL_UPLOAD_PYTHON
  $AltCondaPython = "D:\Programmes\Miniconda3\envs\lgbm-gpu\python.exe"
  if ($OverridePython -and (Test-Path $OverridePython)) {
    Write-Warning "Using MODEL_UPLOAD_PYTHON override: $OverridePython"
    $PythonCmd = $OverridePython
    $PythonArgs = @()
  } elseif (Test-Path $AltCondaPython) {
    Write-Warning "Conda env '$CondaEnvName' not available; using lgbm-gpu python at $AltCondaPython"
    $PythonCmd = $AltCondaPython
    $PythonArgs = @()
  } else {
    $PythonCmd = "python"
    $PythonArgs = @()
  }
}

if (-not (Test-Path $ScriptPath)) {
  throw "Missing script: $ScriptPath"
}

function Ensure-PythonModule {
  param(
    [Parameter(Mandatory = $true)][string]$ModuleName,
    [string]$PipName = $null
  )
  $code = "import importlib,sys;name=sys.argv[1];importlib.import_module(name);sys.exit(0)"
  & $PythonCmd @PythonArgs -c $code $ModuleName
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "Missing required Python module: $ModuleName. Activate your environment or install dependencies."
  }
}

# Ensure core deps for loading models
Ensure-PythonModule -ModuleName "lightgbm"
Ensure-PythonModule -ModuleName "joblib"
Ensure-PythonModule -ModuleName "numpy"
Ensure-PythonModule -ModuleName "pandas"

Write-Host "Building Numerai model upload pickle..."
& $PythonCmd @PythonArgs $ScriptPath --output "model_upload.pkl"
if ($LASTEXITCODE -ne 0) {
  throw "Model upload pickle generation failed with exit code $LASTEXITCODE."
}
Write-Host "Done. Output: $(Join-Path $ProjectDir 'model_upload.pkl')"
