"""Utility helpers for Numerai workflows."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
