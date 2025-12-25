[CmdletBinding()]
param(
    [string]$EnvName = "numerai-env",
    [string]$EnvironmentFile = (Join-Path $PSScriptRoot "numerai-project\environment.yml"),
    [string]$RequirementsFile = (Join-Path $PSScriptRoot "numerai-project\requirements.txt"),
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CondaCommand {
    param(
        [string]$Description,
        [ScriptBlock]$Action
    )
    Write-Host $Description
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed: $Description"
    }
}

function Enable-CondaShell {
    $hook = & conda.exe shell.powershell hook
    if ($LASTEXITCODE -ne 0 -or -not $hook) {
        throw "Unable to initialize conda PowerShell hook."
    }
    Invoke-Expression $hook
}

if (-not (Test-Path $EnvironmentFile)) {
    throw "Environment file not found: $EnvironmentFile"
}

if (-not (Test-Path $RequirementsFile)) {
    throw "Requirements file not found: $RequirementsFile"
}

if (-not (Get-Command conda.exe -ErrorAction SilentlyContinue)) {
    throw "Conda is not available in PATH. Open an Anaconda/Miniconda shell or 'conda init' PowerShell first."
}

$envListJson = (& conda.exe env list --json 2>$null) -join "`n"
if ($LASTEXITCODE -ne 0 -or -not $envListJson) {
    throw "Unable to list conda environments. Ensure conda is initialized."
}
$envPaths = (ConvertFrom-Json $envListJson).envs
$envExists = $envPaths | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }

if ($ForceRecreate -and $envExists) {
    Invoke-CondaCommand "Removing existing env '$EnvName'..." { & conda.exe env remove -n $EnvName -y }
    $envExists = $false
}

if (-not $envExists) {
    Invoke-CondaCommand "Creating conda env '$EnvName' from $EnvironmentFile..." { & conda.exe env create -n $EnvName -f $EnvironmentFile }
} else {
    Invoke-CondaCommand "Updating conda env '$EnvName' from $EnvironmentFile (prune extras)..." { & conda.exe env update -n $EnvName -f $EnvironmentFile --prune }
}

Enable-CondaShell
Write-Host "Activating conda env '$EnvName'..."
conda activate $EnvName
if (-not $?) {
    throw "Failed to activate conda env '$EnvName'."
}

Write-Host "Installing pip requirements from $RequirementsFile..."
python -m pip install -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed for $RequirementsFile"
}

Write-Host "`nDone. Activate with: conda activate $EnvName"
