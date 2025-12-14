"""Utility helpers for Numerai workflows."""
from __future__ import annotations

import datetime as dt
import json
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
    """Keep one canonical key per LightGBM alias group to avoid duplicate warnings."""
    aligned = dict(params or {})
    for primary, alias in LIGHTGBM_ALIAS_GROUPS:
        primary_val = aligned.get(primary)
        alias_val = aligned.get(alias)

        if primary_val is None and alias_val is None:
            continue
        if primary_val is None and alias_val is not None:
            # Promote alias to primary, drop alias
            aligned[primary] = alias_val
            aligned.pop(alias, None)
            continue
        if primary_val is not None and alias_val is None:
            # Only primary present: keep as-is
            continue

        # Both set: keep primary as source of truth, drop alias to silence warnings
        aligned.pop(alias, None)
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


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data to disk (UTF-8)."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def era_slices(era_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique_eras, start_indices, end_indices) for an era-sorted array."""
    if era_values.size == 0:
        empty = np.array([], dtype=str)
        empty_i = np.array([], dtype=int)
        return empty, empty_i, empty_i
    era_values = np.asarray(era_values)
    start_mask = np.empty(era_values.shape[0], dtype=bool)
    start_mask[0] = True
    start_mask[1:] = era_values[1:] != era_values[:-1]
    starts = np.flatnonzero(start_mask)
    unique_eras = era_values[starts]
    ends = np.concatenate([starts[1:], np.array([era_values.shape[0]], dtype=int)])
    return unique_eras, starts, ends


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a safe Pearson correlation (returns 0.0 if variance is 0)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = np.sqrt(np.dot(y_true, y_true)) * np.sqrt(np.dot(y_pred, y_pred))
    if denom == 0.0:
        return 0.0
    return float(np.dot(y_true, y_pred) / denom)


def corr_by_era(era_values: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """Compute Pearson correlation per era (expects era_values aligned with y arrays)."""
    unique_eras, starts, ends = era_slices(np.asarray(era_values))
    corrs = np.empty(unique_eras.shape[0], dtype=np.float64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for i, (s, e) in enumerate(zip(starts, ends)):
        corrs[i] = pearson_corr(y_true[s:e], y_pred[s:e])
    return pd.Series(corrs, index=pd.Index(unique_eras, name="era"), name="corr")


def corr_summary(corrs: pd.Series) -> Dict[str, float]:
    """Summarize a per-era correlation series."""
    corrs = corrs.dropna()
    if corrs.empty:
        return {"mean": float("nan"), "std": float("nan"), "sharpe": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = float(corrs.mean())
    std = float(corrs.std(ddof=0))
    sharpe = float(mean / std) if std > 0 else float("nan")
    return {"mean": mean, "std": std, "sharpe": sharpe, "min": float(corrs.min()), "max": float(corrs.max())}


def rank_uniform(values: pd.Series, era: pd.Series | None = None) -> pd.Series:
    """Rank to a [0, 1] uniform distribution, optionally per-era."""
    if era is None:
        ranked = values.rank(pct=True, method="first")
    else:
        ranked = values.groupby(era, observed=True).rank(pct=True, method="first")
    return ranked.clip(0.0, 1.0)
