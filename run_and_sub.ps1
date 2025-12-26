param(
    [ValidateSet("run", "sub")]
    [string]$Action = "run"
)

$ErrorActionPreference = "Stop"

# Default: enable progress bars unless explicitly disabled by caller
if (-not $env:NUMERAI_PROGRESS) { $env:NUMERAI_PROGRESS = "1" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runScript = Join-Path $ScriptDir "run_me.ps1"
$subScript = Join-Path $ScriptDir "sub_me.ps1"

# Optional .env at repo root (do not override existing env vars)
function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $false }
    $lines = Get-Content -Path $Path
    foreach ($line in $lines) {
        $trim = $line.Trim()
        if (-not $trim) { continue }
        if ($trim.StartsWith("#")) { continue }
        if ($trim.StartsWith("export ")) { $trim = $trim.Substring(7).Trim() }
        $parts = $trim -split "=", 2
        if ($parts.Count -ne 2) { continue }
        $key = $parts[0].Trim()
        if (-not $key) { continue }
        $value = $parts[1].Trim()
        if ($value.Length -ge 2) {
            $doubleQuoted = $value.StartsWith('"') -and $value.EndsWith('"')
            $singleQuoted = $value.StartsWith("'") -and $value.EndsWith("'")
            if ($doubleQuoted -or $singleQuoted) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        if (-not [System.Environment]::GetEnvironmentVariable($key)) {
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    return $true
}

$dotenvPath = Join-Path $ScriptDir ".env"
if (Import-DotEnv -Path $dotenvPath) {
    Write-Host "Loaded env vars from $dotenvPath"
}

if ($Action -eq "run") {
    if (-not (Test-Path $runScript)) {
        Write-Error "Missing run_me.ps1 at $runScript"
        exit 1
    }
    Write-Host "Running full pipeline via run_me.ps1..."
    & $runScript @args
    exit $LASTEXITCODE
}
else {
    if (-not (Test-Path $subScript)) {
        Write-Error "Missing sub_me.ps1 at $subScript"
        exit 1
    }
    Write-Host "Submitting via sub_me.ps1..."
    & $subScript @args
    exit $LASTEXITCODE
}
