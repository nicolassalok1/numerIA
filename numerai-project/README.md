## Numerai – Guide complet (GPU RTX 4060)

Ce fichier est un aide-mémoire détaillé pour relancer l’entraînement/prédiction et la soumission Numerai après plusieurs mois.

### 1. Pré-requis système et environnement
- OS : Windows, PowerShell 7.
- GPU : NVIDIA RTX 4060 (8 Go) avec drivers récents (`nvidia-smi` doit fonctionner).
- Python : environnement conda/micromamba `lgbm-gpu` avec LightGBM compilé CUDA (CUDA toolkit 12.6, compatible runtime 12.9 des drivers).
  - Vérifier l’env : `micromamba run -n lgbm-gpu python -c "import lightgbm, sys; print(lightgbm.__version__, sys.version)"`.
- Stockage : données Numerai dans `data/`, modèles dans `models/`, scripts dans `src/`.

### 2. Scripts clés
- `run_master_autovram.ps1` (PowerShell, à la racine du repo parent) : orchestration complète avec adaptation à la VRAM.
- `run_master.sh` (bash) : orchestration sans auto-VRAM (CPU/GPU selon config).
- `src/train.py` : entraînement (KFold + stacking).
- `src/predict.py` : prédiction et génération de `submission.csv`.

### 3. Pipeline auto-VRAM (recommandé)
Ce mode choisit un tier de paramètres LightGBM en fonction de la VRAM libre et gère le téléchargement des données si absent.

Depuis `numerai-project` :
```powershell
pwsh -File ..\run_master_autovram.ps1
```
Ce que fait le script :
1) Détecte le GPU et lit la VRAM libre via `nvidia-smi`.
2) Utilise les hyperparamètres fixés dans `config/program_input_params.yaml` (source de vérité éditable).
3) Télécharge les datasets Numerai si `data/numerai_training_data.parquet` ou `data/numerai_tournament_data.parquet` manquent.
4) Lance l’entraînement :
   ```powershell
   python src/train.py --config config/training.yaml --params config/program_input_params.yaml --features config/features.yaml
   ```
5) Lance la prédiction :
   ```powershell
   python src/predict.py --config config/training.yaml --params config/program_input_params.yaml --features config/features.yaml
   ```
6) Produit `submission.csv` à la racine du projet (chemin défini dans `config/training.yaml`).

### 4. Pipeline manuel (sans auto-VRAM)
Depuis `numerai-project`, avec les paramètres statiques existants :
```powershell
python src/train.py   --config config/training.yaml --params config/model_params.yaml --features config/features.yaml
python src/predict.py --config config/training.yaml --params config/model_params.yaml --features config/features.yaml
```
Assure-toi que les fichiers `data/numerai_training_data.parquet` et `data/numerai_tournament_data.parquet` sont présents (sinon utiliser `run_master_autovram.ps1` ou `run_master.sh` pour les télécharger).

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
- `nvidia-smi` absent : installer/mettre à jour les drivers NVIDIA ou définir `GPU_REQUIRED=0` (mais perte GPU).
- `git add` échoue sur `.venv/` : ajouter `.venv/` à `.gitignore`.
- VRAM insuffisante : baisser les tiers (safe/medium) ou réduire `row_limit` / `max_features` dans `config/training.yaml` / `config/features.yaml`.

### 7. Références utiles
- `run_master_autovram.ps1` : pipeline complet (check GPU, download data, train/predict) utilisant `config/program_input_params.yaml`.
- `run_master.sh` : équivalent bash sans auto-VRAM.
- `config/program_input_params.yaml` : hyperparamètres LightGBM utilisés par défaut.
- `config/program_output_params.yaml` : emplacement de sortie s’il faut conserver des params générés automatiquement.
- `config/model_params.yaml` : autres paramètres statiques par défaut (GPU).
- `submission.csv` : sortie finale à soumettre.
- Environnement GPU : LightGBM compilé CUDA 12.6 (compatible runtime driver 12.9).
