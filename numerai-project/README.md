## Numerai – Guide complet (GPU RTX 4060)

Ce fichier est un aide-mémoire détaillé pour relancer l’entraînement/prédiction et la soumission Numerai après plusieurs mois.

### 1. Pré-requis système et environnement
- OS : Windows, PowerShell 7.
- GPU : NVIDIA RTX 4060 (8 Go) avec drivers récents (`nvidia-smi` doit fonctionner).
- Python : environnement conda/micromamba `lgbm-gpu` avec LightGBM compilé CUDA (CUDA toolkit 12.6, compatible runtime 12.9 des drivers).
  - Créer/mettre à jour l’environnement GPU : `pwsh -File ..\setup_conda_lightgbm_cuda.ps1 -EnvName lgbm-gpu -PythonVersion 3.11`.
  - Vérifier l’env : `micromamba run -n lgbm-gpu python -c "import lightgbm, sys; print(lightgbm.__version__, sys.version)"`.
  - Test GPU rapide : `python ..\test_lightgbm_gpu.py` (attendu : `device: cuda`).
- Stockage : données Numerai dans `data/`, modèles dans `models/`, scripts dans `src/`.
- Données déjà présentes : `data/numerai_training_data.parquet` et `data/numerai_tournament_data.parquet` doivent exister (le pipeline ne les télécharge pas automatiquement).

### 2. Scripts clés
- `scripts/run_master_autovram.ps1` : orchestration PowerShell (détection du GPU, affichage VRAM, train + predict + upload API).
- `src/train.py` : entraînement (KFold + stacking).
- `src/predict.py` : prédiction et génération de `submission.csv`.
- `config/program_input_params.yaml` : hyperparamètres LightGBM utilisés par défaut.
- `config/features.yaml` / `config/training.yaml` : sélection de features + chemins de fichiers.

### 3. Pipeline scripté (PowerShell)
Ce mode détecte le GPU, affiche la VRAM libre et enchaîne train → predict → upload Numerai à partir des fichiers de config actuels. Il ne télécharge pas les datasets (ils doivent déjà être dans `data/`).

Depuis `numerai-project` :
```powershell
pwsh -File .\scripts\run_master_autovram.ps1
```
Ce que fait le script :
1) Détecte le GPU et lit la VRAM libre via `nvidia-smi` (affichage uniquement).
2) Utilise les hyperparamètres fixés dans `config/program_input_params.yaml` (source de vérité éditable).
3) Entraîne le modèle :
   ```powershell
   python src/train.py --config config/training.yaml --params config/program_input_params.yaml --features config/features.yaml
   ```
4) Lance la prédiction :
   ```powershell
   python src/predict.py --config config/training.yaml --params config/program_input_params.yaml --features config/features.yaml
   ```
5) Soumet `submission.csv` via l’API Numerai avec les variables d’environnement présentes dans la session PowerShell (`NUMERAI_PUBLIC_ID`, `NUMERAI_SECRET_KEY`, `NUMERAI_MODEL_ID`).

Avant de lancer le script, définis les secrets dans ta session (éviter de les laisser dans le fichier) :
```powershell
$env:NUMERAI_PUBLIC_ID="VRAI_PUBLIC_ID"
$env:NUMERAI_SECRET_KEY="VRAIE_SECRET_KEY"
$env:NUMERAI_MODEL_ID="VRAI_MODEL_ID"
```

### 4. Pipeline manuel (sans auto-VRAM)
Depuis `numerai-project`, avec les paramètres statiques existants (LightGBM GPU agressif) :
```powershell
python src/train.py   --config config/training.yaml --params config/program_input_params.yaml --features config/features.yaml
python src/predict.py --config config/training.yaml --params config/program_input_params.yaml --features config/features.yaml
```
Les modèles entraînés sont déposés dans `models/` et la soumission dans `submission.csv` (chemins définis dans `config/training.yaml`). Assure-toi que les fichiers `data/numerai_training_data.parquet` et `data/numerai_tournament_data.parquet` sont présents.

### 5. Soumission Numerai via l’API `numerapi`
Définir les secrets dans la session PowerShell (copiés depuis le dashboard Numerai) :
```powershell
$env:NUMERAI_PUBLIC_ID="VRAI_PUBLIC_ID"
$env:NUMERAI_SECRET_KEY="VRAIE_SECRET_KEY"
$env:NUMERAI_MODEL_ID="VRAI_MODEL_ID"
```
Soumettre `submission.csv` :
```powershell
@'
from numerapi import NumerAPI
import os
napi = NumerAPI(os.environ["NUMERAI_PUBLIC_ID"], os.environ["NUMERAI_SECRET_KEY"])
resp = napi.upload_predictions("submission.csv", model_id=os.environ["NUMERAI_MODEL_ID"])
print("Upload status:", resp)
'@ | python -
```
Le retour doit contenir un ID de submission. Vérifier ensuite le statut sur le dashboard ou via l’API.

### 6. Dépannage rapide
- `session is invalid or expired` : Secret Key ou Public ID incorrects/expirés, ou Model ID non associé au compte. Regénérer la Secret Key et vérifier le Model ID.
- Données absentes : le pipeline ne télécharge pas. Récupérer les fichiers Numerai officiels et placer `numerai_training_data.parquet` et `numerai_tournament_data.parquet` dans `data/`.
- `nvidia-smi` absent : installer/mettre à jour les drivers NVIDIA ou définir `GPU_REQUIRED=0` (mais perte GPU).
- `git add` échoue sur `.venv/` : ajouter `.venv/` à `.gitignore`.
- VRAM insuffisante : baisser les tiers (safe/medium) ou réduire `row_limit` / `max_features` dans `config/training.yaml` / `config/features.yaml`.
- Paramètres manquants : le projet n’utilise plus `config/model_params.yaml`; passer `--params config/program_input_params.yaml` à `train.py` et `predict.py`.

### 7. Références utiles
- `scripts/run_master_autovram.ps1` : pipeline complet (train + predict + upload) utilisant `config/program_input_params.yaml`.
- `config/program_input_params.yaml` : hyperparamètres LightGBM (tiers agressif par défaut).
- `config/features.yaml` / `config/training.yaml` : sélection des features et chemins des fichiers.
- `submission.csv` : sortie finale à soumettre.
- `setup_conda_lightgbm_cuda.ps1` : build/installation de LightGBM CUDA dans un env conda.
- Environnement GPU : LightGBM compilé CUDA 12.6 (compatible runtime driver 12.9).
