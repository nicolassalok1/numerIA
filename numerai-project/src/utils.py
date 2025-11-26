"""Utility helpers for Numerai workflows."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

TARGET_CANDIDATES = [
    "target",
    "target_kazutsugi",
    "target_cyrus_v4",
    "target_nomi",
    "target_jericho",
]

# Keep alias pairs in sync to avoid LightGBM noise when both names are present
LIGHTGBM_ALIAS_GROUPS = [
    ("feature_fraction", "colsample_bytree"),
    ("bagging_fraction", "subsample"),
    ("bagging_freq", "subsample_freq"),
    ("min_data_in_leaf", "min_child_samples"),
    ("min_sum_hessian_in_leaf", "min_child_weight"),
    ("lambda_l1", "reg_alpha"),
    ("lambda_l2", "reg_lambda"),
]


def log(message: str) -> None:
    """Minimal console logger with timestamp."""
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} UTC] {message}")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file safely."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_params(params_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize various YAML layouts into a unified params dict.

    Supports current repo files where LightGBM params live under `model`
    (or at top-level) and MLP params can be top-level as well.
    """
    params_cfg = params_cfg or {}
    normalized: Dict[str, Any] = {"lightgbm": {}, "ridge": {}, "mlp": {}, "stacker": {}}

    lightgbm_keys = {
        "boosting_type",
        "objective",
        "metric",
        "device_type",
        "gpu_platform_id",
        "gpu_device_id",
        "max_depth",
        "num_leaves",
        "min_data_in_leaf",
        "min_sum_hessian_in_leaf",
        "max_bin",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "learning_rate",
        "n_estimators",
        "verbosity",
        "lambda_l1",
        "lambda_l2",
        "min_gain_to_split",
    }
    mlp_keys = {
        "hidden_layer_sizes",
        "layers",
        "learning_rate_init",
        "alpha",
        "batch_size",
        "max_iter",
        "early_stopping",
        "n_iter_no_change",
        "validation_fraction",
        "tol",
    }

    lightgbm_params = params_cfg.get("lightgbm") or params_cfg.get("model")
    ridge_params = params_cfg.get("ridge") or {}
    stacker_params = params_cfg.get("stacker") or {}
    mlp_params = params_cfg.get("mlp")

    mlp_candidates = {k: v for k, v in params_cfg.items() if k in mlp_keys}
    has_mlp_like = bool(mlp_params) or bool(mlp_candidates)

    if not lightgbm_params:
        lgb_candidates = {k: v for k, v in params_cfg.items() if k in lightgbm_keys}
        if lgb_candidates:
            lightgbm_params = lgb_candidates
        elif params_cfg and not has_mlp_like and not any(k in params_cfg for k in ("ridge", "stacker", "lightgbm", "model")):
            lightgbm_params = dict(params_cfg)

    if not mlp_params and mlp_candidates:
        mlp_params = mlp_candidates

    normalized["lightgbm"] = lightgbm_params or {}
    normalized["ridge"] = ridge_params or {}
    normalized["mlp"] = mlp_params or {}
    normalized["stacker"] = stacker_params or {}
    return normalized


def align_lightgbm_aliases(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure LightGBM alias parameters share the same value to silence warnings."""
    aligned = dict(params or {})
    for primary, alias in LIGHTGBM_ALIAS_GROUPS:
        primary_val = aligned.get(primary)
        alias_val = aligned.get(alias)
        if primary_val is not None and alias_val is None:
            aligned[alias] = primary_val
        elif alias_val is not None and primary_val is None:
            aligned[primary] = alias_val
        elif primary_val is not None and alias_val is not None and primary_val != alias_val:
            # Keep primary as source of truth and mirror to alias to avoid "ignored" warnings
            aligned[alias] = primary_val
    return aligned


def parquet_columns(path: str | Path) -> List[str]:
    """Return parquet column names without loading the dataset."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception as exc:  # pragma: no cover - schema probe is best-effort
        log(f"Unable to inspect schema for {path}: {exc}")
        return []


def safe_read_parquet(path: str | Path, columns: List[str] | None = None) -> pd.DataFrame:
    """Read parquet file safely, returning an empty frame if missing or failing."""
    path = Path(path)
    if not path.exists():
        log(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:  # pragma: no cover - IO failures are best-effort
        log(f"Failed to read parquet {path}: {exc}")
        return pd.DataFrame()


def get_feature_columns(df: pd.DataFrame, prefix: str = "feature") -> List[str]:
    """Return list of feature columns matching the given prefix."""
    return [c for c in df.columns if c.startswith(prefix)]


def find_target_column(columns: Iterable[str]) -> str | None:
    """Find the first known target column name in a list."""
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def detect_target(df: pd.DataFrame) -> str:
    """Detect a target column name from common Numerai targets."""
    return find_target_column(df.columns) or "target"


def dummy_dataset(n_rows: int = 100, n_features: int = 3, prefix: str = "feature") -> Tuple[pd.DataFrame, pd.Series]:
    """Create a small dummy dataset to keep scripts runnable without data."""
    data = np.arange(n_rows * n_features, dtype=float).reshape(n_rows, n_features)
    cols = [f"{prefix}{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    target = pd.Series(np.zeros(n_rows, dtype=float), name="target")
    return df, target


def save_submission(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save predictions to CSV in Numerai format."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    predictions.to_csv(output_path, index=False)
    log(f"Saved submission: {output_path}")
    return output_path
