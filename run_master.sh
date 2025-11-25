#!/usr/bin/env bash
set -euo pipefail

# Détection du dossier projet (fonctionne si le script est à la racine ou dans le dossier parent).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/config" && -d "${SCRIPT_DIR}/src" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/numerai-project/config" ]]; then
  ROOT_DIR="${SCRIPT_DIR}/numerai-project"
else
  echo "[ERREUR] Impossible de localiser le projet (config/ et src/ attendus)." >&2
  exit 1
fi
cd "$ROOT_DIR"

echo "Projet : $ROOT_DIR"

###############################################################################
# 1) Vérifs de base
###############################################################################
command -v python3 >/dev/null || { echo "python3 manquant"; exit 1; }
command -v numerai >/dev/null && HAS_NUMERAI=1 || HAS_NUMERAI=0

# GPU requis (LightGBM configuré en device=gpu)
GPU_REQUIRED="${GPU_REQUIRED:-1}"
if command -v nvidia-smi >/dev/null; then
  echo "GPU détecté :"
  nvidia-smi || true
else
  if [[ "$GPU_REQUIRED" -eq 1 ]]; then
    echo "[ERREUR] GPU requis mais nvidia-smi introuvable. Installez les drivers NVIDIA ou définissez GPU_REQUIRED=0 pour forcer le mode CPU." >&2
    exit 1
  else
    echo "[AVERTISSEMENT] GPU non détecté, exécution CPU (GPU_REQUIRED=0)."
  fi
fi

###############################################################################
# 2) Téléchargement des données Numerai
###############################################################################
python3 - <<'PY'
from pathlib import Path
import numerapi

data_dir = Path("data")
train_src = data_dir / "numerai_training_data.parquet"
tour_src  = data_dir / "numerai_tournament_data.parquet"
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

###############################################################################
# 3) Entraînement + Prédiction (configs du repo)
###############################################################################
echo "=> Entraînement..."
python3 src/train.py --config config/training.yaml --params config/model_params.yaml --features config/features.yaml

echo "=> Prédiction..."
python3 src/predict.py --config config/training.yaml --params config/model_params.yaml --features config/features.yaml
echo "Submission générée (cf. config/training.yaml, par défaut submission.csv)."

###############################################################################
# 4) Docker (optionnel) : DOCKER=1 ./run_master.sh
###############################################################################
if [[ "${DOCKER:-0}" == "1" ]]; then
  echo "=> Build Docker..."
  docker build -t numerai:latest docker
  echo "=> Run Docker (montage data/ et models/)..."
  docker run --rm -v "$ROOT_DIR/data:/app/data" -v "$ROOT_DIR/models:/app/models" numerai:latest
fi

###############################################################################
# 5) GitHub Actions / secrets (rappel)
###############################################################################
cat <<'INFO'
---
GitHub Actions : .github/workflows/numerai_pipeline.yml
Secrets requis dans le repo :
  - NUMERAI_PUBLIC_ID
  - NUMERAI_SECRET_KEY
  - NUMERAI_MODEL_ID
Soumission CI : numerai submit --model-id $NUMERAI_MODEL_ID --public-id $NUMERAI_PUBLIC_ID --secret-key $NUMERAI_SECRET_KEY submission.csv
INFO

###############################################################################
# 6) Soumission locale (si numerai-cli installé et secrets en env)
###############################################################################
if [[ "$HAS_NUMERAI" -eq 1 && -n "${NUMERAI_PUBLIC_ID:-}" && -n "${NUMERAI_SECRET_KEY:-}" && -n "${NUMERAI_MODEL_ID:-}" ]]; then
  echo "=> Soumission locale via numerai-cli..."
  numerai submit \
    --model-id "$NUMERAI_MODEL_ID" \
    --public-id "$NUMERAI_PUBLIC_ID" \
    --secret-key "$NUMERAI_SECRET_KEY" \
    submission.csv || true
else
  echo "Soumission locale ignorée (numerai-cli ou secrets manquants)."
fi

echo "Terminé."
