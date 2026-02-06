$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $RootDir "numerai-project"
$ScriptPath = Join-Path $ProjectDir "src\build_upload_pkls.py"

if (-not (Test-Path $ScriptPath)) {
  throw "Missing script: $ScriptPath"
}

$PythonCmd = $env:MODEL_UPLOAD_PYTHON
if (-not $PythonCmd) {
  $PythonCmd = "D:\Programmes\Miniconda3\envs\lgbm-gpu\python.exe"
}
if (-not (Test-Path $PythonCmd)) {
  $PythonCmd = "python"
}

Write-Host "Building Numerai upload pkls (hello_numerai, feature_neutralization, target_ensemble)..."
& $PythonCmd $ScriptPath --out-dir "."
if ($LASTEXITCODE -ne 0) {
  throw "build_upload_pkls failed with exit code $LASTEXITCODE."
}

Write-Host "Done. Output files in numerai-project\."
