
You are working on a Numerai project in VS Code on Windows, using conda and PowerShell 7.

Existing project structure:
- config/
- src/
- data/
- models/

Python entrypoints (already implemented and working):
- src/train.py
- src/predict.py

Both `train.py` and `predict.py` accept:
  --config   config/training.yaml
  --params   <params yaml path>
  --features config/features.yaml

The LightGBM model runs on GPU (device_type=gpu) on an NVIDIA RTX 4060 (8GB VRAM) with 64GB system RAM and an i7 CPU.

I want you to CREATE a new PowerShell 7 script at the project root called:

  run_master_autovram.ps1

This script must implement an **auto-VRAM adaptive** behavior as follows.

====================================================================
0) General setup
====================================================================

- At the top of the script, set:
  `$ErrorActionPreference = "Stop"`

- Determine the project root directory `$RootDir` based on the script’s location:

  ```powershell
  $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

  if (Test-Path (Join-Path $ScriptDir "config") -and (Test-Path (Join-Path $ScriptDir "src"))) {
      $RootDir = $ScriptDir
  }
  elseif (Test-Path (Join-Path $ScriptDir "numerai-project\config")) {
      $RootDir = Join-Path $ScriptDir "numerai-project"
  }
  else {
      Write-Error "[ERROR] Cannot locate project root (config/ and src/ expected)."
      exit 1
  }

  Set-Location $RootDir
  Write-Host "Project root: $RootDir"
````

* Check that `python` is available in PATH using `Get-Command python -ErrorAction SilentlyContinue`.

  * If not found, print a clear error and exit.

* Check if the `numerai` CLI is installed using `Get-Command numerai -ErrorAction SilentlyContinue`.

  * Set `$HAS_NUMERAI = 1` if found, otherwise `0`.

====================================================================

1. GPU detection + VRAM query
   ====================================================================

* If `$env:GPU_REQUIRED` is not set, default it to `"1"`.

* Detect `nvidia-smi`:

  ```powershell
  $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  ```

* If `nvidia-smi` **is found**:

  * Print "GPU detected:" and call `nvidia-smi` once to display the GPU state.

  * Then query total and free VRAM:

    ```powershell
    $vramInfo = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
    ```

  * Assume a single GPU line like: `8192, 7340`

  * Parse it into two integers:

    * `$totalVramMb`
    * `$freeVramMb`

  * Print, for example:
    `Write-Host "VRAM total: $totalVramMb MB, free: $freeVramMb MB"`

* If `nvidia-smi` is **not found**:

  * If `$env:GPU_REQUIRED -eq "1"`:

    * `Write-Error` a message that GPU is required but `nvidia-smi` is missing, then exit.
  * Else (GPU_REQUIRED = "0"):

    * `Write-Warning` that GPU is not detected and continue (we will still generate a config, but you can mention that it’s risky / CPU fallback).

====================================================================
2) Auto-VRAM based LightGBM model configuration
===============================================

* The script must automatically select one **tier** of hyperparameters for LightGBM based on `$freeVramMb`.

* Define 4 tiers based on free VRAM (you can implement in a simple if/elseif/else chain):

  * If `$freeVramMb -ge 7000`  → tier `"aggressive"`
  * ElseIf `$freeVramMb -ge 5000` → tier `"high"`
  * ElseIf `$freeVramMb -ge 3000` → tier `"medium"`
  * Else                         → tier `"safe"`

* For each tier, define a specific set of hyperparameters:

  Common fields for ALL tiers:

  ```yaml
  model:
    boosting_type: gbdt
    objective: regression
    metric: mae

    device_type: gpu
    gpu_platform_id: 0
    gpu_device_id: 0

    max_depth: -1
    min_sum_hessian_in_leaf: 1e-3
    lambda_l1: 0.0
    lambda_l2: 5.0
    min_gain_to_split: 0.0
    verbosity: -1
  ```

  Then override these for each tier:

  * aggressive:
    num_leaves: 512
    max_bin: 511
    learning_rate: 0.02
    n_estimators: 2500
    feature_fraction: 0.9
    bagging_fraction: 0.9
    bagging_freq: 1
    min_data_in_leaf: 60

  * high:
    num_leaves: 256
    max_bin: 511
    learning_rate: 0.03
    n_estimators: 2000
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 1
    min_data_in_leaf: 100

  * medium:
    num_leaves: 128
    max_bin: 255
    learning_rate: 0.05
    n_estimators: 1500
    feature_fraction: 0.7
    bagging_fraction: 0.7
    bagging_freq: 1
    min_data_in_leaf: 150

  * safe:
    num_leaves: 64
    max_bin: 255
    learning_rate: 0.05
    n_estimators: 1000
    feature_fraction: 0.6
    bagging_fraction: 0.6
    bagging_freq: 1
    min_data_in_leaf: 200

* Once the tier is chosen, build a YAML string in PowerShell representing:

  ```yaml
  model:
    ...
  ```

  with all the fields for the selected tier, and write it to:

  `config/model_params_autovram.yaml`

  Use something like `Set-Content -Path "config/model_params_autovram.yaml" -Encoding UTF8`.

* Print a summary, for example:

  ```powershell
  Write-Host "Selected auto-VRAM tier: $tier (free VRAM: $freeVramMb MB)"
  Write-Host "Using params file: config/model_params_autovram.yaml"
  ```

====================================================================
3) Numerai data download (if missing)
=====================================

* Check if both files exist:

  * `data/numerai_training_data.parquet`
  * `data/numerai_tournament_data.parquet`

* If they both exist, print:

  `"Datasets already present, skipping download."`

* Otherwise, embed an inline Python script in a here-string that:

  * Uses `from pathlib import Path` and `import numerapi`
  * Creates `data/` if needed.
  * Uses `napi = numerapi.NumerAPI()` and `datasets = napi.list_datasets()`
  * Defines a helper to pick dataset names ending with certain suffixes and excluding names that contain "benchmark", "example", or "pred".
  * Picks:

    * training dataset from suffixes:
      ["train.parquet", "numerai_training_data.parquet", "latest_numerai_training_data.parquet"]
    * tournament dataset from suffixes:
      ["live.parquet", "numerai_tournament_data.parquet", "latest_numerai_tournament_data.parquet"]
  * Downloads them to:
    data/numerai_training_data.parquet
    data/numerai_tournament_data.parquet

* Write this inline Python to a temp file in `$env:TEMP`, run it with `python`, then delete the temp file.

====================================================================
4) Training and prediction
==========================

* Once `config/model_params_autovram.yaml` exists and data are available, run training:

  ```powershell
  & python "src/train.py" `
      --config   "config/training.yaml" `
      --params   "config/model_params_autovram.yaml" `
      --features "config/features.yaml"
  ```

* If the training call fails (exception or non-zero `$LASTEXITCODE`), print a clear error and exit.

* If training succeeds, run prediction:

  ```powershell
  & python "src/predict.py" `
      --config   "config/training.yaml" `
      --params   "config/model_params_autovram.yaml" `
      --features "config/features.yaml"
  ```

* At the end, print a final summary along the lines of:

  * Selected tier
  * Free VRAM used for the decision
  * Params file path (config/model_params_autovram.yaml)
  * Confirmation that submission.csv has been generated (assuming training.yaml defines that).

====================================================================
5) Optional numerai CLI submission
==================================

* If `$HAS_NUMERAI -eq 1` **and** all of the following environment variables are non-empty:

  * `$env:NUMERAI_PUBLIC_ID`
  * `$env:NUMERAI_SECRET_KEY`
  * `$env:NUMERAI_MODEL_ID`

  then attempt an automatic submission:

  ```powershell
  try {
      numerai submit `
        --model-id  $env:NUMERAI_MODEL_ID `
        --public-id $env:NUMERAI_PUBLIC_ID `
        --secret-key $env:NUMERAI_SECRET_KEY `
        "submission.csv"
  }
  catch {
      Write-Warning "Numerai CLI submission failed: $($_.Exception.Message)"
  }
  ```

* Otherwise, print a message stating that local submission is skipped (either numerai CLI or secrets are missing).

====================================================================
Implementation requirements
===========================

* Use idiomatic PowerShell 7, with proper indentation.
* Use `Write-Host` for normal info, `Write-Warning` for non-fatal issues, `Write-Error` for fatal ones.
* Do not implement anything beyond what is specified above.
* Output the **full content** of run_master_autovram.ps1.

```
```
