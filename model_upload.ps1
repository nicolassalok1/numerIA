$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $RootDir "numerai-project"
$ScriptPath = Join-Path $ProjectDir "src\model_upload.py"

if (-not (Test-Path $ScriptPath)) {
  throw "Missing script: $ScriptPath"
}

Write-Host "Building Numerai model upload pickle..."
python $ScriptPath --output "model_upload.pkl"
if ($LASTEXITCODE -ne 0) {
  throw "Model upload pickle generation failed with exit code $LASTEXITCODE."
}
Write-Host "Done. Output: $(Join-Path $ProjectDir 'model_upload.pkl')"
