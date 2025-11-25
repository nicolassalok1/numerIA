"""Utility helpers for Numerai workflows."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd


def log(message: str) -> None:
    """Minimal console logger with timestamp."""
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} UTC] {message}")


def safe_read_parquet(path: str | Path) -> pd.DataFrame:
    """Read parquet file safely, returning an empty frame if missing."""
    path = Path(path)
    if not path.exists():
        log(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    return pd.read_parquet(path)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_submission(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    """Save predictions to CSV in Numerai format."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    predictions.to_csv(output_path, index=False)
    log(f"Saved submission: {output_path}")
    return output_path
