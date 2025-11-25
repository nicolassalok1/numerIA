# Numerai Machine Learning Workspace

Workspace preconfigure pour experimenter, entrainer et automatiser des modeles Numerai. La structure suit une separation claire entre configuration, code, donnees et automatisation.

## Structure du projet
- `config/` : fichiers YAML pour les features, l'entrainement et les hyperparametres modeles.
- `data/` : jeux de donnees locaux (ignore par git).
- `models/` : artefacts et submissions (ignore par git).
- `notebooks/` : prototypage rapide et EDA.
- `src/` : logique Python pour entrainer, predire et empiler les modeles.
- `automation/` : scripts utilitaires et exemple de cron pour soumettre automatiquement.
- `environment.yml` / `requirements.txt` : dependances Conda ou pip.
- `numerai.code-workspace` : workspace VS Code preconfigure.

## Installation de l'environnement
```bash
# Option Conda
conda env create -f environment.yml
conda activate numerai-env

# Option pip (dans un venv)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrainement
```bash
python src/train.py \
  --config config/training.yaml \
  --params config/model_params.yaml
```
Le script charge les hyperparametres (LightGBM, Ridge, MLP) et entraine les modeles definis. Les sorties et artefacts sont sauvegardes dans `models/` selon la configuration.

## Prediction
```bash
python src/predict.py --params config/model_params.yaml
```
La sortie est un DataFrame de predictions (colonne `prediction`). Utilisez `automation/submit.py` pour generer un CSV pret a soumettre.

## Automatisation des submissions
- Le script `automation/submit.py` cree un fichier CSV dans `models/submission.csv` a partir des predictions.
- Le fichier `automation/cron_example.txt` fournit une ligne cron exemple :
  - `0 9 * * MON /usr/bin/python3 /home/user/numerai-project/automation/submit.py`
- Pour l'activer, ajoutez la ligne dans `crontab -e` en ajustant les chemins vers votre environnement et script Python.

## Ouverture rapide du workspace VS Code
```bash
code numerai.code-workspace
```
Cette commande peut etre ajoutee a un alias (ex: `alias numeraiws="code /home/user/numerai-project/numerai.code-workspace"`) ou appelee via Copilot Chat pour ouvrir le workspace directement.

Commande Copilot prete a l'emploi :
```bash
./copilot_open_workspace.sh
```
Copilot Chat peut aussi executer ce script pour ouvrir le workspace sans saisir manuellement la commande.


























---

Crée un projet complet Numerai dans un workspace VS Code.
Je veux **tous les fichiers**, **tous les dossiers**, **toutes les configs**, et **toute l’automatisation**, organisés proprement selon les instructions ci-dessous.

Génère **automatiquement** :

* l’arborescence complète,
* tous les fichiers,
* les templates,
* les configs,
* le workflow GitHub Actions,
* le Dockerfile,
* l’environnement Conda,
* la tâche VS Code pour lancer les trainings,
* les secrets instructions,
* et le script d’automatisation.

Le projet doit s’appeler :

```
numerai-project/
```

---

# 📁 1) ARBORESCENCE COMPLÈTE DU PROJET

Crée exactement cette structure :

```
numerai-project/
│
├── config/
│   ├── features.yaml
│   ├── model_params.yaml
│   └── training.yaml
│
├── data/
│   ├── .gitkeep
│
├── models/
│   ├── .gitkeep
│
├── notebooks/
│   ├── numerai_pipeline.ipynb
│   ├── EDA.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   ├── model_lgb.py
│   ├── model_ridge.py
│   ├── model_mlp.py
│   └── stacker.py
│
├── automation/
│   ├── submit.py
│   └── cron_example.txt
│
├── .github/
│   └── workflows/
│       └── numerai_pipeline.yml
│
├── docker/
│   ├── Dockerfile
│   └── .dockerignore
│
├── .vscode/
│   ├── tasks.json
│   ├── keybindings.json
│
├── environment.yml
├── requirements.txt
├── README.md
├── .gitignore
└── numerai.code-workspace
```

---

# 📌 2) `.gitignore`

Ignore :

```
__pycache__/
*.pkl
*.parquet
.env
models/
data/
.ipynb_checkpoints/
```

---

# 📌 3) FICHIERS DE CONFIG :

## `config/features.yaml`

```
features:
  prefix: "feature"
```

## `config/model_params.yaml`

Exact :

```
lightgbm:
  boosting_type: "gbdt"
  n_estimators: 1200
  learning_rate: 0.01
  num_leaves: 63
  feature_fraction: 0.8
  bagging_fraction: 0.8
  reg_alpha: 1.0
  reg_lambda: 1.0
  metric: "rmse"

ridge:
  alpha: 1.0

mlp:
  layers: [256,128,64]
  alpha: 1e-5
  learning_rate_init: 1e-3
  max_iter: 30

stacker:
  alpha: 0.5
```

## `config/training.yaml`

```
general:
  seed: 42
  n_folds: 5

files:
  train: "data/numerai_training_data.parquet"
  tournament: "data/numerai_tournament_data.parquet"
  submission: "submission.csv"
```

---

# 📌 4) FICHIERS PYTHON

Pour chaque fichier Python, créer :

* imports
* docstring
* squelette minimal fonctionnel
* fonctions placeholders

Fichiers :
`train.py`, `predict.py`, `utils.py`, `model_lgb.py`, `model_ridge.py`, `model_mlp.py`, `stacker.py`, `__init__.py`.

Dans `train.py`, utilise **KFold stacking** comme ici :

* LightGBM
* Ridge
* MLP
* KFold 5 folds
* Stacker = Ridge(alpha=0.5)
* sauvegarde des modèles dans `models/`

---

# 📌 5) NOTEBOOKS

Créer :

### `notebooks/numerai_pipeline.ipynb`

→ version notebook de la pipeline

### `notebooks/EDA.ipynb`

→ analyse rapide des features et target

---

# 📌 6) AUTOMATION

## `automation/submit.py`

→ doit exécuter `predict.py`.

## `automation/cron_example.txt`

```
0 9 * * MON /usr/bin/python3 /home/user/numerai-project/automation/submit.py
```

---

# 📌 7) WORKFLOW GITHUB ACTIONS

Créer `.github/workflows/numerai_pipeline.yml` :

Fonctionnalités :

* schedule hebdo (lundi 10:00 UTC)
* workflow_dispatch
* checkout repo
* setup python 3.10
* installer requirements
* installer numerapi
* download Numerai data
* train model
* predict
* upload submission avec :

  * `${{ secrets.NUMERAI_PUBLIC_ID }}`
  * `${{ secrets.NUMERAI_SECRET_KEY }}`
  * `${{ secrets.NUMERAI_MODEL_ID }}`

Upload via :

```
numerai submit \
  --model-id $NUMERAI_MODEL_ID \
  --public-id $NUMERAI_PUBLIC_ID \
  --secret-key $NUMERAI_SECRET_KEY \
  submission.csv
```

---

# 📌 8) DOCKER

Dossier : `docker/`

### `docker/Dockerfile`

Basé sur python:3.10-slim
Installe requirements + numerapi
CMD = `python3 src/predict.py`

### `.dockerignore`

```
__pycache__/
*.pkl
*.parquet
data/
models/
.ipynb_checkpoints/
```

---

# 📌 9) VS CODE INTEGRATION

### `.vscode/tasks.json`

Créer une tâche `"Train Numerai Model"` qui :

```
conda activate numerai-env && python3 src/train.py
```

### `.vscode/keybindings.json`

Lier le training à :

```
Ctrl+Shift+T
```

---

# 📌 10) CONDA ENV

Créer un fichier `environment.yml` :

```
name: numerai-env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - lightgbm
  - pyyaml
  - pip
  - pip:
      - joblib
      - numerapi
```

---

# 📌 11) requirements.txt

```
pandas
numpy
scikit-learn
lightgbm
pyyaml
joblib
numerapi
```

---

# 📌 12) README.md professionnel

Décrire :

* structure du projet
* comment entraîner (`python3 src/train.py`)
* comment prédire (`python3 src/predict.py`)
* comment utiliser cron
* comment utiliser GitHub Actions
* comment utiliser Docker

---

# 📌 13) Workspace VS Code

Créer `numerai.code-workspace` avec :

* ouverture automatique de :

  * src/
  * notebooks/
  * config/
  * automation/

Ensuite : **ouvrir automatiquement ce workspace** après génération.

