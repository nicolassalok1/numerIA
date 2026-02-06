# README-AWS.md

Date: 2026-02-06

This file summarizes the AWS cloud setup and automation work done in this repo, plus how to reproduce it on a new machine after a fresh clone.

**What we built today**
- 3 Numerai Prediction Nodes on AWS using 
umerai-cli.
- One node per tournament:
  - Classic (tournament 8): model salok1_classic
  - Signals (tournament 11): model salok1_signals
  - Crypto (tournament 12): model salok1
- Dedicated node folders with Dockerfiles and predict.py for each tournament:
  - 
umerai-project\classic-node
  - 
umerai-project\signals-node
  - 
umerai-project\crypto-node
- Daily automation scripts to train, build pickles, deploy, and test.
- A health check script to verify all three nodes with one command.

**How cloud automation works**
- After 
umerai node config + 
umerai node deploy + 
umerai node test, Numerai calls the webhook automatically every day.
- If a test passes once, daily submissions are automatic with no manual action.

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
`powershell
git clone <YOUR_REPO_URL>
cd numerIA
`

### 3) Plug-and-play setup (local environment)
`powershell
pwsh -File .\setup_plug_and_play.ps1
`
Optional flags:
- -SkipGpuBuild to skip building GPU LightGBM
- -SkipNumeraiCli to skip numerai-cli install
- -SkipDockerCheck to skip Docker checks
- -CreateKeysTemplate to create keys_local.template.ps1

### 4) Add local keys (for local submissions)
Create keys_local.ps1 in the repo root:
`powershell
="YOUR_PUBLIC_ID"
="YOUR_SECRET_KEY"
="YOUR_MODEL_ID"
`

### 5) Configure Numerai CLI + AWS
`powershell
numerai setup --provider aws
`
This writes config under %USERPROFILE%\.numerai\.

### 6) Configure nodes + deploy + test
`powershell
pwsh -File .\daily_full_setup.ps1
`
This will:
- configure nodes if they do not exist
- train locally
- build pickles
- deploy nodes
- run tests

### 7) Daily automation (Task Scheduler)
You can schedule the full pipeline:
`powershell
pwsh -File .\daily_full_setup.ps1 -ScheduleTime 02:30
`

---

## Troubleshooting notes (AWS / Numerai CLI)

### Terraform error: compute_environment_name not expected
Fix in:
- %USERPROFILE%\.numerai\aws\aws\cluster.tf
Replace compute_environment_name = ... with:
- 
ame = local.node_prefix

### Terraform error: Most Recent Image Not Filtered
Fix in:
- %USERPROFILE%\.numerai\aws\aws\cluster.tf
Add owners = ["amazon"] to the data "aws_ami" "ecs_al2" block.

### Windows encoding errors during deploy/test
Set UTF-8 output in PowerShell before running 
umerai node:
`powershell
="1"
="utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
=""; =""; =""
`

---

## Tournament nodes and local artifacts

**Classic (t8)**
- Node folder: 
umerai-project\classic-node
- Pickle file: 
umerai-project\classic-node\salok1_classic.pkl

**Signals (t11)**
- Node folder: 
umerai-project\signals-node
- Pickle file: 
umerai-project\signals-node\signals.pkl
- Local sample data (optional): D:\PythonDProjects\numerIA\signals\v2.1\SAMPLES

**Crypto (t12)**
- Node folder: 
umerai-project\crypto-node
- Pickle file: 
umerai-project\crypto-node\salok1.pkl

---

## PowerShell scripts (use cases)

Root scripts:
- uild_upload_pkls.ps1 : Builds hello_numerai.pkl, eature_neutralization.pkl, 	arget_ensemble.pkl from trained models.
- daily_full_setup.ps1 : Config + train + build pickles + deploy + test for Classic, Signals, Crypto.
- daily_retrain_deploy.ps1 : Train + build pickles + deploy only (no config).
- health_check.ps1 : Runs 
umerai node test for Classic, Signals, Crypto in one command.
- model_upload.ps1 : Builds model_upload.pkl from trained models (classic upload format).
- un_and_sub.ps1 : Local pipeline (un_me.ps1) or just submit (sub_me.ps1), reads keys_local.ps1.
- un_classic_v5.ps1 : Classic training + build salok1_classic.pkl + copy to classic node.
- un_crypto_v2.ps1 : Crypto training + build salok1.pkl + copy to crypto node.
- un_me.ps1 : Local full classic training + predict + upload using NumerAPI.
- un_signals_v2.ps1 : Signals training + build signals.pkl + copy to signals node.
- setup_numerai_env.ps1 : Creates 
umerai-env conda env from 
umerai-project\environment.yml.
- sub_me.ps1 : Local submission only (reads keys_local.ps1).
- setup_plug_and_play.ps1 : One-command environment setup for a new machine.
- keys_local.ps1 : Local secrets file (do not commit real secrets).

Project scripts:
- 
umerai-project\scripts\setup_conda_lightgbm_cuda.ps1 : Builds LightGBM with CUDA into the lgbm-gpu conda env.

LightGBM build CI helpers (not used by Numerai pipeline):
- lightgbm_gpu_build\.ci\install-opencl.ps1
- lightgbm_gpu_build\.ci\lint-powershell.ps1
- lightgbm_gpu_build\.ci\test-r-package-windows.ps1
- lightgbm_gpu_build\.ci\test-windows.ps1

---

## Reproducing the cloud setup on another machine

1) Run setup_plug_and_play.ps1 to install local tools and build the GPU environment.
2) Run 
umerai setup --provider aws to configure the CLI and credentials.
3) Update model names in daily_full_setup.ps1 if your models differ.
4) Run daily_full_setup.ps1 to config, deploy, and test all nodes.
5) Confirm tests pass. Numerai will submit daily automatically after that.
