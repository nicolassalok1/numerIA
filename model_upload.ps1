$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $RootDir "numerai-project"
$ScriptPath = Join-Path $ProjectDir "src\model_upload.py"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$CondaPython = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { $null }
$CondaEnvName = if ($env:MODEL_UPLOAD_CONDA_ENV) { $env:MODEL_UPLOAD_CONDA_ENV } else { "numerai-env" }
$EnvFile = Join-Path $ProjectDir "environment.yml"
if ($CondaPython -and (Test-Path $CondaPython)) {
  $PythonExe = $CondaPython
  $PythonRunner = @($PythonExe)
} elseif (Test-Path $VenvPython) {
  $PythonExe = $VenvPython
  $PythonRunner = @($PythonExe)
} else {
  $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
  if ($condaCmd) {
    if (Test-Path $EnvFile) {
      Write-Host "Updating conda env '$CondaEnvName' from $EnvFile"
      conda env update -n $CondaEnvName -f $EnvFile
      if ($LASTEXITCODE -ne 0) {
        throw "Conda env update failed for $CondaEnvName."
      }
    }
    $PythonExe = "conda run -n $CondaEnvName python"
    $PythonRunner = @("conda", "run", "-n", $CondaEnvName, "python")
  } else {
    $PythonExe = "python"
    $PythonRunner = @($PythonExe)
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
  $code = @"
import importlib, sys
name = sys.argv[1]
try:
    importlib.import_module(name)
    sys.exit(0)
except Exception:
    sys.exit(1)
"@
  $code | & $PythonRunner - $ModuleName
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "Missing required Python module: $ModuleName. Activate your environment or install dependencies."
  }
}

# Ensure core deps for loading models
Ensure-PythonModule -ModuleName "lightgbm"
Ensure-PythonModule -ModuleName "cloudpickle"
Ensure-PythonModule -ModuleName "joblib"
Ensure-PythonModule -ModuleName "numpy"
Ensure-PythonModule -ModuleName "pandas"

Write-Host "Building Numerai model upload pickle..."
& $PythonRunner $ScriptPath --output "model_upload.pkl"
if ($LASTEXITCODE -ne 0) {
  throw "Model upload pickle generation failed with exit code $LASTEXITCODE."
}
Write-Host "Done. Output: $(Join-Path $ProjectDir 'model_upload.pkl')"
