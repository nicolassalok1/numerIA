#!/usr/bin/env bash
set -euo pipefail

# Script allégé pour éviter les OOM : échantillonne le training, réduit les folds et les hyperparams.
# GPU requis par défaut (LightGBM device=gpu). Définir GPU_REQUIRED=0 pour ignorer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_REQUIRED="${GPU_REQUIRED:-1}"
if command -v nvidia-smi >/dev/null; then
  echo "GPU détecté :"
  nvidia-smi || true
else
  if [[ "$GPU_REQUIRED" -eq 1 ]]; then
    echo "[ERREUR] GPU requis mais nvidia-smi introuvable. Définissez GPU_REQUIRED=0 pour forcer le mode CPU." >&2
    exit 1
  else
    echo "[AVERTISSEMENT] GPU non détecté, exécution CPU (GPU_REQUIRED=0)."
  fi
fi

TRAIN_SRC="data/numerai_training_data.parquet"
TOUR_SRC="data/numerai_tournament_data.parquet"
SAMPLE_PATH="data/train_sample.parquet"
TMP_TRAIN_CFG="$(mktemp)"
TMP_PARAMS_CFG="$(mktemp)"

echo "Projet : $ROOT_DIR"

autodl() {
python3 - <<'PY'
from pathlib import Path
import numerapi

train_src = Path("data/numerai_training_data.parquet")
tour_src = Path("data/numerai_tournament_data.parquet")
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

if train_src.exists() and tour_src.exists():
    print("Datasets déjà présents, skip download.")
    raise SystemExit

napi = numerapi.NumerAPI()
datasets = napi.list_datasets()

def pick_dataset(candidates, exclude=None):
    exclude = exclude or []
    for suffix in candidates:
        matches = sorted(
            d for d in datasets
            if d.endswith(suffix)
            and not any(ex.lower() in d.lower() for ex in exclude)
        )
        if matches:
            return matches[-1]
    raise SystemExit(f"Aucun dataset trouvé pour {candidates}. Exemples: {datasets[:5]}")

train_ds = pick_dataset(
    ["train.parquet", "numerai_training_data.parquet", "latest_numerai_training_data.parquet"],
    exclude=["benchmark", "example", "pred"]
)
tour_ds = pick_dataset(
    ["live.parquet", "numerai_tournament_data.parquet", "latest_numerai_tournament_data.parquet"],
    exclude=["benchmark", "example", "pred"]
)

print(f"Téléchargement train: {train_ds}")
napi.download_dataset(train_ds, str(train_src))
print(f"Téléchargement tournoi: {tour_ds}")
napi.download_dataset(tour_ds, str(tour_src))
print("Données téléchargées dans", data_dir.resolve())
PY
}

autosample() {
python3 - <<'PY'
from pathlib import Path
import pandas as pd

train_src = Path("data/numerai_training_data.parquet")
sample_path = Path("data/train_sample.parquet")

if sample_path.exists():
    print("Échantillon déjà présent, skip.")
    raise SystemExit

df = pd.read_parquet(train_src)
n_rows = min(200_000, len(df))
sample = df.sample(n=n_rows, random_state=42)
sample.to_parquet(sample_path, index=False)
print(f"Échantillon écrit: {sample_path} ({len(sample)} lignes)")
PY
}

# Téléchargement minimal si nécessaire
autodl || true

# Échantillonnage pour limiter la RAM
autosample || true

# Configs temporaires allégées
cat > "$TMP_TRAIN_CFG" <<'YAML'
general:
  seed: 42
  n_folds: 2

files:
  train: "data/train_sample.parquet"
  tournament: "data/numerai_tournament_data.parquet"
  submission: "submission.csv"
YAML

cat > "$TMP_PARAMS_CFG" <<'YAML'
lightgbm:
  boosting_type: "gbdt"
  n_estimators: 400
  learning_rate: 0.05
  num_leaves: 31
  max_depth: 6
  feature_fraction: 0.6
  bagging_fraction: 0.6
  bagging_freq: 1
  min_data_in_leaf: 500
  reg_alpha: 1.0
  reg_lambda: 1.0
  metric: "rmse"
  device: "gpu"
  gpu_platform_id: 0
  gpu_device_id: 0

ridge:
  alpha: 1.0

mlp:
  layers: [128, 64]
  alpha: 1e-4
  learning_rate_init: 1e-3
  max_iter: 100

stacker:
  alpha: 0.5
YAML

export TMP_TRAIN_CFG TMP_PARAMS_CFG SAMPLE_PATH

echo "=> Entraînement (échantillon, 2 folds)..."
python3 src/train.py --config "$TMP_TRAIN_CFG" --params "$TMP_PARAMS_CFG" --features config/features.yaml

echo "=> Prédiction..."
python3 src/predict.py --config "$TMP_TRAIN_CFG" --params "$TMP_PARAMS_CFG" --features config/features.yaml

echo "Submission générée (chemin : $(python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open('$TMP_TRAIN_CFG'))
print(cfg['files']['submission'])
PY))"

echo "Configs temporaires : $TMP_TRAIN_CFG / $TMP_PARAMS_CFG (supprimez-les après usage si besoin)."
