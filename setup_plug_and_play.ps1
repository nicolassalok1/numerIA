param(
    [switch],
    [switch],
    [switch],
    [switch]
)

Continue = "Stop"

 = Split-Path -Parent System.Management.Automation.InvocationInfo.MyCommand.Path
Set-Location 

Write-Host "=== NumerAI plug-and-play setup ==="
Write-Host "Repo: "

function Write-Step {
    param([string])
    Write-Host "
>>> "
}

Write-Step "Checking conda availability"
 = Get-Command conda -ErrorAction SilentlyContinue
if (-not ) {
    Write-Error "conda not found in PATH. Install Miniconda/Anaconda, then re-run."
    exit 1
}

if (-not ) {
    Write-Step "Ensuring GPU LightGBM env (lgbm-gpu)"
     = &  env list | Select-String -Pattern "^\s*lgbm-gpu\s"
    if (-not ) {
         = Join-Path  "numerai-project\scripts\setup_conda_lightgbm_cuda.ps1"
        if (-not (Test-Path )) {
            Write-Error "Missing GPU setup script: "
            exit 1
        }
        &  -EnvName "lgbm-gpu" -PythonVersion "3.11"
    } else {
        Write-Host "Conda env 'lgbm-gpu' already exists."
    }
} else {
    Write-Host "Skipping GPU LightGBM build (SkipGpuBuild)."
}

if (-not ) {
    Write-Step "Checking numerai CLI"
     = Get-Command numerai -ErrorAction SilentlyContinue
    if (-not ) {
        Write-Host "numerai CLI not found. Attempting install with pip --user."
         = Get-Command python -ErrorAction SilentlyContinue
        if () {
            & .Source -m pip install --upgrade numerai-cli --user
        } else {
            Write-Warning "python not found in PATH; install numerai-cli manually: pip install --upgrade numerai-cli --user"
        }
    } else {
        Write-Host "numerai CLI found: "
    }
} else {
    Write-Host "Skipping numerai CLI check/install (SkipNumeraiCli)."
}

if (-not ) {
    Write-Step "Checking Docker"
     = Get-Command docker -ErrorAction SilentlyContinue
    if (-not ) {
        Write-Warning "Docker not found in PATH. Install Docker Desktop to deploy cloud nodes."
    } else {
        try {
            & .Source version | Out-Null
            Write-Host "Docker OK."
        } catch {
            Write-Warning "Docker is installed but not running. Start Docker Desktop."
        }
    }
}

if () {
    Write-Step "Creating keys_local.template.ps1"
     = Join-Path  "keys_local.template.ps1"
    if (-not (Test-Path )) {
@'
="YOUR_PUBLIC_ID"
="YOUR_SECRET_KEY"
="YOUR_MODEL_ID"
'@ | Set-Content -Encoding UTF8 
        Write-Host "Created "
    } else {
        Write-Host "Template already exists: "
    }
}

Write-Step "Next steps"
Write-Host "1) Configure Numerai CLI and AWS credentials: numerai setup --provider aws"
Write-Host "2) Create models on Numerai dashboard and update script model names if needed"
Write-Host "3) Run daily_full_setup.ps1 to config/deploy/test all nodes"
Write-Host "4) Schedule daily_full_setup.ps1 via Task Scheduler if desired"

Write-Host "Setup complete."
