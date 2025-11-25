#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
GRID_FOLDS="2 3"
GRID_EST="300 500"
GRID_LEAVES="31 63"
GRID_MAXBIN="255 511"

echo "Grid search léger (échantillon) depuis $ROOT_DIR"
python3 - <<'PY'  # génère l'échantillon si absent
from pathlib import Path
import pandas as pd
train_src = Path("data/numerai_training_data.parquet")
sample_path = Path("data/train_sample.parquet")
if not sample_path.exists():
    df = pd.read_parquet(train_src)
    n = min(200_000, len(df))
    df.sample(n=n, random_state=42).to_parquet(sample_path, index=False)
    print(f"Échantillon écrit: {n} lignes -> {sample_path}")
else:
    print("Échantillon déjà présent, on continue.")
PY

mkdir -p tmp_grid_logs
LOGFILE="tmp_grid_logs/grid_log.txt"
: > "$LOGFILE"

for F in $GRID_FOLDS; do
  for NE in $GRID_EST; do
    for NL in $GRID_LEAVES; do
      for MB in $GRID_MAXBIN; do
        CFG_TRAIN="$(mktemp)"
        CFG_PARAMS="$(mktemp)"
        cat > "$CFG_TRAIN" <<EOF
general:
  seed: 42
  n_folds: $F
files:
  train: "data/train_sample.parquet"
  tournament: "data/numerai_tournament_data.parquet"
  submission: "submission.csv"
EOF
        cat > "$CFG_PARAMS" <<EOF
lightgbm:
  boosting_type: "gbdt"
  n_estimators: $NE
  learning_rate: 0.05
  num_leaves: $NL
  max_depth: 6
  max_bin: $MB
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
EOF
        echo "Test folds=$F est=$NE leaves=$NL max_bin=$MB"
        START=$(date +%s)
        if python3 src/train.py --config "$CFG_TRAIN" --params "$CFG_PARAMS" --features config/features.yaml >/dev/null 2>&1; then
          STATUS=OK
        else
          STATUS=FAIL
        fi
        END=$(date +%s)
        DURATION=$((END-START))
        echo "folds=$F est=$NE leaves=$NL max_bin=$MB status=$STATUS time=${DURATION}s" | tee -a "$LOGFILE"
        rm -f "$CFG_TRAIN" "$CFG_PARAMS"
      done
    done
  done
done

echo "Terminé. Résultats dans $LOGFILE"
