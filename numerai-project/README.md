## Numerai – Guide complet (GPU RTX 4060)

Aide-mémoire pour l'entraînement, la prédiction et la soumission Numerai.

### 1. Pré-requis système et environnement
- OS : Windows, PowerShell 7.
- GPU : NVIDIA RTX 4060 (8 Go) avec drivers récents (`nvidia-smi` doit fonctionner).
- Python : environnement conda `lgbm-gpu` avec LightGBM compilé CUDA (CUDA toolkit 12.6).
  - Créer/mettre à jour l'environnement GPU : `pwsh -File scripts\setup_conda_lightgbm_cuda.ps1 -EnvName lgbm-gpu -PythonVersion 3.11`.
  - Test GPU rapide : `python scripts\test_lightgbm_gpu.py` (attendu : `device: cuda`).
- Stockage : données dans `data/`, modèles dans `models/`, scripts dans `src/`.

### 2. Modèles et nodes AWS

| Tournament | Modèle | Node folder | Pickle |
|---|---|---|---|
| Classic (t8) | salok1_classic | `classic-node/` | `salok1_classic.pkl` |
| Signals (t11) | salok1_signals | `signals-node/` | `signals.pkl` |
| Crypto (t12) | salok1 | `crypto-node/` | `salok1.pkl` |
| Classic (t8) | tgrv2 | `tgrv2-node/` | `tgrv2.pkl` |

Chaque node contient un `Dockerfile`, `predict.py`, `requirements.txt` et le pickle du modèle. Le webhook Numerai déclenche la soumission automatiquement chaque jour.

### 3. Scripts clés
- `run_me.ps1` (racine `numerIA`) : orchestration PowerShell (détection GPU, VRAM, train + predict + upload API).
- `src/train.py` : entraînement (KFold + stacking).
- `src/predict.py` : prédiction et génération de `submission.csv`.
- `config/program_input_params.yaml` : hyperparamètres LightGBM.
- `config/features.yaml` / `config/training.yaml` : sélection de features + chemins de fichiers.

### 4. Pipeline scripté (PowerShell)

Depuis `numerIA` (racine) :
```powershell
pwsh -File .\run_me.ps1
```
Ce que fait le script :
1. Détecte le GPU et lit la VRAM libre via `nvidia-smi`.
2. Utilise les hyperparamètres de `config/program_input_params.yaml`.
3. Entraîne le modèle via `src/train.py`.
4. Lance la prédiction via `src/predict.py`.
5. Soumet `submission.csv` via l'API Numerai.

Avant de lancer, charger les secrets via `keys_local.ps1` (racine, git-ignored) :
```powershell
$env:NUMERAI_PUBLIC_ID="..."
$env:NUMERAI_SECRET_KEY="..."
$env:NUMERAI_MODEL_ID="..."
```

Le script rafraîchit les datasets sauf si `SKIP_DATA_REFRESH=1`.

### 5. Scripts par tournoi
- `run_classic_v5.ps1` : Classic training + `salok1_classic.pkl` -> `classic-node/`
- `run_signals_v2.ps1` : Signals training + `signals.pkl` -> `signals-node/`
- `run_crypto_v2.ps1` : Crypto training + `salok1.pkl` -> `crypto-node/`

### 6. Automatisation quotidienne

```powershell
# Config + train + deploy + test des 4 nodes
pwsh -File .\daily_full_setup.ps1

# Train + deploy uniquement (sans config ni test)
pwsh -File .\daily_retrain_deploy.ps1

# Health check des 4 nodes
pwsh -File .\health_check.ps1

# Planification quotidienne via Task Scheduler
pwsh -File .\daily_full_setup.ps1 -ScheduleTime 02:30
```

### 7. Dépannage rapide
- `session is invalid or expired` : Secret Key ou Public ID incorrects/expirés. Regénérer depuis le dashboard.
- `invalid_submission_ids` : le live set est obsolète. Laisser `run_me.ps1` rafraîchir les datasets.
- `numerai` non trouvé dans PATH : utiliser le chemin complet `C:\Users\nicol\AppData\Roaming\Python\Python313\Scripts\numerai.exe`.
- Docker non lancé : démarrer Docker Desktop avant `numerai node config/deploy/test`.
- Terraform state corrompu (batch job queue) : voir `README-AWS.md` section Troubleshooting.
- VRAM insuffisante : réduire `row_limit` / `max_features` dans les fichiers de config.
