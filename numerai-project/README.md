# Numerai ML Workspace

Workspace pret a l'emploi pour experimenter, entrainer, predire et automatiser des soumissions Numerai avec VS Code, GitHub Actions et Docker.

## Structure
- `config/` : YAML (features, hyperparametres, chemins de fichiers).
- `data/` : donnees locales (ignorees), `.gitkeep` pour versionner le dossier.
- `models/` : artefacts et submissions (ignorees), `.gitkeep` fourni.
- `notebooks/` : `numerai_pipeline.ipynb` et `EDA.ipynb` pour prototypage.
- `src/` : code Python (entrainement KFold LightGBM/Ridge/MLP, prediction, utilitaires, stacker Ridge).
- `automation/` : script de soumission et exemple de cron.
- `.github/workflows/` : pipeline CI de telechargement, entrainement, prediction, soumission.
- `docker/` : Dockerfile et .dockerignore pour execution conteneurisee.
- `.vscode/` : tache de training + raccourci clavier.
- `environment.yml` / `requirements.txt` : dependances Conda ou pip.
- `numerai.code-workspace` : workspace VS Code preconfigure.

## Installation
```bash
# Conda
conda env create -f environment.yml
conda activate numerai-env

# Ou pip (dans un venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrainement (KFold + stacker)
```bash
python3 src/train.py \
  --config config/training.yaml \
  --params config/model_params.yaml \
  --features config/features.yaml
```
- Utilise LightGBM, Ridge, MLP en 5 folds (configurable) et stacker Ridge(alpha=0.5).
- Sauvegarde les modeles dans `models/` via joblib.

## Prediction
```bash
python3 src/predict.py \
  --config config/training.yaml \
  --params config/model_params.yaml \
  --features config/features.yaml
```
- Charge les modeles sauves (ou placeholders si absents), genere un DataFrame `prediction` et ecrit le CSV `submission.csv` (chemin dans `config/training.yaml`).

## Automatisation locale
- Script : `python3 automation/submit.py` (utilise les chemins par defaut des configs) pour produire une submission prete.
- Cron exemple (`automation/cron_example.txt`) :
  - `0 9 * * MON /usr/bin/python3 /home/user/numerai-project/automation/submit.py`
  - Adaptez le chemin vers votre Python/env, puis ajoutez dans `crontab -e`.

## GitHub Actions
- Workflow : `.github/workflows/numerai_pipeline.yml`.
- Actions : checkout, install deps, numerapi/numerai-cli, telechargement des donnees, train, predict, submit.
- Secrets requis dans le repo :
  - `NUMERAI_PUBLIC_ID`
  - `NUMERAI_SECRET_KEY`
  - `NUMERAI_MODEL_ID`
- Submission via `numerai submit --model-id $NUMERAI_MODEL_ID ... submission.csv`.

## Docker
```bash
cd docker
docker build -t numerai-predict ..
docker run --rm -v "$(pwd)/../data:/app/data" -v "$(pwd)/../models:/app/models" numerai-predict
```
- Image base `python:3.10-slim`, installe les requirements + numerapi, CMD `python3 src/predict.py`.

## VS Code
- Workspace : `code numerai.code-workspace` ou `./copilot_open_workspace.sh`.
- Tache : `Train Numerai Model` lance `conda activate numerai-env && python3 src/train.py`.
- Raccourci : `Ctrl+Shift+T` declenche la tache de training (voir `.vscode/keybindings.json`).

## Donnees
- Le workflow CI telecharge automatiquement les parquet Numerai via numerapi.
- En local, lancez le snippet CI ou telechargez via `numerapi.NumerAPI().download_dataset(...)` pour peupler `data/`.
