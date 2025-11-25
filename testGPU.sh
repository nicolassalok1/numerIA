#!/usr/bin/env bash
set -euo pipefail

echo "=== ENVIRONNEMENT PYTHON ==="
echo "Conda env : ${CONDA_DEFAULT_ENV:-<inconnu>}"
echo "Python    : $(which python)"

echo
echo "=== GPU via nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi introuvable (driver NVIDIA non détecté dans cet env)."
fi

echo
echo "=== Test LightGBM avec device='gpu' ==="
python - <<'PY'
import numpy as np

try:
    import lightgbm as lgb
except Exception as e:
    print("LightGBM non importable :", e)
else:
    print("LightGBM version :", lgb.__version__)
    print("Config device_type :", lgb.basic._ConfigAliases().get("device_type", "unknown"))

    X = np.random.rand(10000, 10)
    y = np.random.rand(10000)

    try:
        model = lgb.LGBMRegressor(
            device="gpu",
            n_estimators=10,
            max_depth=4,
            num_leaves=31,
        )
        model.fit(X, y)
        print(">>> GPU LightGBM OK : entraînement terminé avec device='gpu'.")
    except Exception as e:
        print(">>> GPU LightGBM KO :", repr(e))
PY

echo
echo "=== FIN DU TEST GPU PYTHON ==="
