param(
    [string]$EnvName = "lgbm-gpu",
    [string]$PythonVersion = "3.11",
    [string]$LgbmSrcDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $LgbmSrcDir) {
    $LgbmSrcDir = Join-Path (Get-Location) "lightgbm_gpu_build"
}

# Force l'usage de conda-forge et évite les ToS Anaconda
$CondaForgeArgs = @("-c", "conda-forge", "--override-channels")

# Récupère l'exécutable conda réel (et évite la fonction PowerShell cassée)
$condaExe = $env:CONDA_EXE
if (-not $condaExe -or -not (Test-Path $condaExe)) {
    $condaCmd = Get-Command conda -CommandType Application -ErrorAction SilentlyContinue
    if ($condaCmd) {
        $condaExe = $condaCmd.Source
    }
}

if (-not $condaExe -or -not (Test-Path $condaExe)) {
    Write-Error "Erreur : impossible de trouver l'exécutable 'conda'. Vérifie l'installation Anaconda/Miniconda."
    exit 1
}

Write-Host ">>> Création/initialisation de l'environnement conda '$EnvName' (Python $PythonVersion)"

# Vérifie si l'environnement existe déjà
$envExists = & $condaExe env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $envExists) {
    & $condaExe create -y -n $EnvName "python=$PythonVersion" @CondaForgeArgs
} else {
    Write-Host ">>> L'environnement '$EnvName' existe déjà, il sera utilisé."
}

Write-Host ">>> Installation des dépendances pour LightGBM + CUDA dans l'environnement '$EnvName'"

& $condaExe install -y -n $EnvName @CondaForgeArgs `
    cmake `
    ninja `
    numpy `
    scipy `
    scikit-learn `
    pandas `
    cudatoolkit

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host ">>> 'git' non trouvé, installation dans l'environnement conda..."
    & $condaExe install -y -n $EnvName @CondaForgeArgs git
}

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Warning "ATTENTION : 'nvcc' (CUDA toolkit) n'est pas dans le PATH. Le build GPU peut échouer si le toolkit NVIDIA CUDA n'est pas installé sur la machine."
}

Write-Host ">>> Récupération du code source de LightGBM dans '$LgbmSrcDir'"

if (-not (Test-Path $LgbmSrcDir)) {
    & $condaExe run -n $EnvName git clone --recursive https://github.com/microsoft/LightGBM.git $LgbmSrcDir
} else {
    Write-Host ">>> Le dossier '$LgbmSrcDir' existe déjà, mise à jour du dépôt..."
    & $condaExe run -n $EnvName git -C $LgbmSrcDir pull
    & $condaExe run -n $EnvName git -C $LgbmSrcDir submodule update --init --recursive
}

$buildDir = Join-Path $LgbmSrcDir "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Write-Host ">>> Compilation de LightGBM avec support CUDA (GPU)"

Push-Location $buildDir
& $condaExe run -n $EnvName cmake .. -DUSE_GPU=ON -DUSE_CUDA=ON -DUSE_OPENCL=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja
& $condaExe run -n $EnvName cmake --build . --config Release
Pop-Location

Write-Host ">>> Installation du package Python LightGBM (lié à la version GPU compilée)"

$pythonPkgDir = Join-Path $LgbmSrcDir "python-package"
# scikit-build-core attend un fichier LICENSE à côté du pyproject
$licenseSrc = Join-Path $LgbmSrcDir "LICENSE"
if (Test-Path $licenseSrc) {
    Copy-Item $licenseSrc (Join-Path $pythonPkgDir "LICENSE") -Force
}
& $condaExe run -n $EnvName python -m pip install -U pip setuptools wheel scikit-build-core ninja
& $condaExe run -n $EnvName python -m pip install -e $pythonPkgDir `
    --config-settings=cmake.define.USE_CUDA=ON `
    --config-settings=cmake.define.USE_GPU=ON `
    --config-settings=cmake.define.USE_OPENCL=OFF `
    --config-settings=cmake.source-dir=$LgbmSrcDir

Write-Host ""
Write-Host ">>> Terminé."
Write-Host "Active ensuite l'environnement avec :"
Write-Host "    conda activate $EnvName"
Write-Host "Puis dans Python, tu peux tester le GPU avec :"
Write-Host "    import lightgbm as lgb"
Write-Host "    clf = lgb.LGBMClassifier(device='gpu')"
