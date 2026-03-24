# README-AWS.md

Date: 2026-03-24

This file summarizes the AWS cloud setup and automation for this repo, plus how to reproduce it on a new machine.

## Cloud nodes

4 Numerai Prediction Nodes on AWS using numerai-cli:

| Tournament | Node name | Model name | Model ID | Node folder |
|---|---|---|---|---|
| Classic (t8) | numerai-salok1_classic | salok1_classic | `c3a95af2-...` | `numerai-project\classic-node` |
| Signals (t11) | signals-salok1_signals | salok1_signals | `d8c538bd-...` | `numerai-project\signals-node` |
| Crypto (t12) | crypto-salok1 | salok1 | `ea40cb4a-...` | `numerai-project\crypto-node` |
| Classic (t8) | numerai-tgrv2 | tgrv2 | `f35f60cf-...` | `numerai-project\tgrv2-node` |

Each node folder contains: `Dockerfile`, `predict.py`, `requirements.txt`, and a `.pkl` model file.

## How cloud automation works

After `numerai node config` + `numerai node deploy` + `numerai node test`, Numerai calls the webhook automatically every day. If a test passes once, daily submissions are automatic with no manual action.

---

## Installation on a new computer (fresh git clone)

### 1) Prerequisites
- Windows + PowerShell 7
- Git
- Docker Desktop (running)
- Miniconda/Anaconda (conda in PATH)
- NVIDIA drivers + CUDA toolkit if you want GPU training
- A Numerai account with API keys and models
- AWS account for compute nodes

### 2) Clone the repo
```powershell
git clone <YOUR_REPO_URL>
cd numerIA
```

### 3) Plug-and-play setup (local environment)
```powershell
pwsh -File .\setup_plug_and_play.ps1
```
Optional flags:
- `-SkipGpuBuild` to skip building GPU LightGBM
- `-SkipNumeraiCli` to skip numerai-cli install
- `-SkipDockerCheck` to skip Docker checks
- `-CreateKeysTemplate` to create `keys_local.template.ps1`

### 4) Add local keys (for local submissions)
Create `keys_local.ps1` in the repo root (git-ignored):
```powershell
$env:NUMERAI_PUBLIC_ID="YOUR_PUBLIC_ID"
$env:NUMERAI_SECRET_KEY="YOUR_SECRET_KEY"
$env:NUMERAI_MODEL_ID="YOUR_MODEL_ID"
```

### 5) Configure Numerai CLI + AWS
```powershell
numerai setup --provider aws
```
This writes config under `%USERPROFILE%\.numerai\`.

### 6) Configure nodes + deploy + test
```powershell
pwsh -File .\daily_full_setup.ps1
```
This will:
- configure nodes if they do not exist (Classic, Signals, Crypto, TGRV2)
- train locally
- build pickles
- deploy nodes
- run tests

### 7) Daily automation (Task Scheduler)
```powershell
pwsh -File .\daily_full_setup.ps1 -ScheduleTime 02:30
```

---

## Troubleshooting notes (AWS / Numerai CLI)

### Terraform error: compute_environment_name not expected
Fix in `%USERPROFILE%\.numerai\aws\aws\cluster.tf`:
Replace `compute_environment_name = ...` with `name = local.node_prefix`.

### Terraform error: Most Recent Image Not Filtered
Fix in `%USERPROFILE%\.numerai\aws\aws\cluster.tf`:
Add `owners = ["amazon"]` to the `data "aws_ami" "ecs_al2"` block.

### Terraform error: Missing Resource Identity After Read (batch job queue)
The Batch job queue was deleted outside Terraform. Fix:
```powershell
docker run --rm -it -v $env:USERPROFILE\.numerai:/opt/plan -w /opt/plan -e "AWS_ACCESS_KEY_ID=..." -e "AWS_SECRET_ACCESS_KEY=..." hashicorp/terraform:1.5.6 -chdir=aws state rm 'module.aws[0].aws_batch_job_queue.node'
```
Then re-run `numerai node config`.

### Windows encoding errors during deploy/test
Set UTF-8 output in PowerShell before running numerai node:
```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""
```

### numerai CLI not in PATH
Use the full path:
```powershell
& "C:\Users\nicol\AppData\Roaming\Python\Python313\Scripts\numerai.exe" node ...
```

---

## Tournament nodes and local artifacts

**Classic (t8) - salok1_classic**
- Node folder: `numerai-project\classic-node`
- Pickle file: `numerai-project\classic-node\salok1_classic.pkl`

**Signals (t11) - salok1_signals**
- Node folder: `numerai-project\signals-node`
- Pickle file: `numerai-project\signals-node\signals.pkl`

**Crypto (t12) - salok1**
- Node folder: `numerai-project\crypto-node`
- Pickle file: `numerai-project\crypto-node\salok1.pkl`

**Classic (t8) - tgrv2**
- Node folder: `numerai-project\tgrv2-node`
- Pickle file: `numerai-project\tgrv2-node\tgrv2.pkl`

---

## PowerShell scripts

Root scripts:
- `daily_full_setup.ps1` : Config + train + build pickles + deploy + test for all 4 nodes.
- `daily_retrain_deploy.ps1` : Train + build pickles + deploy only (no config).
- `health_check.ps1` : Runs `numerai node test` for all 4 nodes in one command.
- `run_classic_v5.ps1` : Classic training + build `salok1_classic.pkl` + copy to classic node.
- `run_signals_v2.ps1` : Signals training + build `signals.pkl` + copy to signals node.
- `run_crypto_v2.ps1` : Crypto training + build `salok1.pkl` + copy to crypto node.
- `run_me.ps1` : Local full classic training + predict + upload using NumerAPI.
- `run_and_sub.ps1` : Local pipeline (`run_me.ps1`) or just submit (`sub_me.ps1`).
- `sub_me.ps1` : Local submission only (reads `keys_local.ps1`).
- `build_upload_pkls.ps1` : Builds pickle files from trained models.
- `model_upload.ps1` : Builds `model_upload.pkl` from trained models (classic upload format).
- `setup_plug_and_play.ps1` : One-command environment setup for a new machine.
- `setup_numerai_env.ps1` : Creates `numerai-env` conda env from `numerai-project\environment.yml`.
- `keys_local.ps1` : Local secrets file (git-ignored, do not commit).

Project scripts:
- `numerai-project\scripts\setup_conda_lightgbm_cuda.ps1` : Builds LightGBM with CUDA into the `lgbm-gpu` conda env.

---

## Reproducing the cloud setup on another machine

1. Run `setup_plug_and_play.ps1` to install local tools and build the GPU environment.
2. Run `numerai setup --provider aws` to configure the CLI and credentials.
3. Update model names in `daily_full_setup.ps1` if your models differ.
4. Run `daily_full_setup.ps1` to config, deploy, and test all nodes.
5. Confirm tests pass. Numerai will submit daily automatically after that.
