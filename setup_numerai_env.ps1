$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $RootDir "numerai-project"
$EnvFile = Join-Path $ProjectDir "environment.yml"
$EnvName = "numerai-env"

if (-not (Test-Path $EnvFile)) {
  throw "Missing environment.yml at $EnvFile"
}

$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
  throw "conda not found on PATH. Please install Miniconda/Anaconda and retry."
}

Write-Host "Setting up conda env '$EnvName' from $EnvFile"

conda --no-plugins env list | Select-String -SimpleMatch $EnvName | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "Env exists. Updating..."
  conda --no-plugins env update -n $EnvName -f $EnvFile --solver classic
} else {
  Write-Host "Env not found. Creating..."
  conda --no-plugins env create -n $EnvName -f $EnvFile
}

if ($LASTEXITCODE -ne 0) {
  throw "Conda env setup failed for $EnvName."
}

Write-Host "Done. Activate with: conda activate $EnvName"
