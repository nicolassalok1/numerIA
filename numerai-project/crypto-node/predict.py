"""Numerai Crypto prediction script (loads a pre-trained pickle)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import joblib
import numerapi
import numpy as np
import pandas as pd

logging.basicConfig(filename="log.txt", filemode="a", level=logging.INFO)

TOURNAMENT = 12
DATA_VERSION = os.getenv("NUMERAI_CRYPTO_VERSION", "crypto/v2.0")
LIVE_DATASET = os.getenv("NUMERAI_CRYPTO_LIVE_DATASET", "").strip()
LIVE_FILENAME = os.getenv("NUMERAI_CRYPTO_LIVE_FILE", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", "salok1.pkl")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH", "predictions.csv")
RANK_UNIFORM = os.getenv("NUMERAI_RANK_UNIFORM", "1").lower() not in {"0", "false", "no"}

DEFAULT_MODEL_ID = None
DEFAULT_PUBLIC_ID = None
DEFAULT_SECRET_KEY = None

MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)

napi = numerapi.NumerAPI(
    public_id=os.getenv("NUMERAI_PUBLIC_ID", DEFAULT_PUBLIC_ID),
    secret_key=os.getenv("NUMERAI_SECRET_KEY", DEFAULT_SECRET_KEY),
)


def _safe_load_model(path: str | Path) -> Any:
    model_path = Path(path)
    try:
        return joblib.load(model_path)
    except Exception as exc:
        logging.warning("joblib.load failed: %s", exc)
        data = model_path.read_bytes()
        try:
            import cloudpickle as serializer  # type: ignore
        except Exception:
            import pickle as serializer  # type: ignore
    return serializer.loads(data)


def _download_first_available(datasets: Sequence[str]) -> str:
    for dataset in datasets:
        if not dataset:
            continue
        try:
            logging.info("Attempting download: %s", dataset)
            napi.download_dataset(dataset)
            return dataset
        except Exception as exc:
            logging.warning("Failed to download %s: %s", dataset, exc)
    raise RuntimeError("Failed to download any live dataset. Check NUMERAI_CRYPTO_LIVE_DATASET.")


def _maybe_extract_features(model: Any) -> List[str] | None:
    if isinstance(model, dict):
        for key in ("feature_cols", "feature_columns", "features"):
            cols = model.get(key)
            if isinstance(cols, (list, tuple)) and cols and all(isinstance(c, str) for c in cols):
                return list(cols)
    for attr in ("feature_cols", "feature_columns", "feature_names_in_"):
        cols = getattr(model, attr, None)
        if isinstance(cols, (list, tuple, np.ndarray)):
            cols_list = [str(c) for c in list(cols)]
            return cols_list if cols_list else None
    for attr in ("feature_name_",):
        cols = getattr(model, attr, None)
        if isinstance(cols, (list, tuple)) and cols:
            return [str(c) for c in cols]
    feature_name_fn = getattr(model, "feature_name", None)
    if callable(feature_name_fn):
        try:
            cols = feature_name_fn()
            if isinstance(cols, (list, tuple)) and cols:
                return [str(c) for c in cols]
        except Exception:
            pass
    booster = getattr(model, "booster_", None)
    if booster is not None:
        feature_name_fn = getattr(booster, "feature_name", None)
        if callable(feature_name_fn):
            try:
                cols = feature_name_fn()
                if isinstance(cols, (list, tuple)) and cols:
                    return [str(c) for c in cols]
            except Exception:
                pass
    return None


def _unwrap_model_and_features(model: Any) -> tuple[Any, List[str] | None]:
    if isinstance(model, dict) and "model" in model:
        return model.get("model"), _maybe_extract_features(model)
    if isinstance(model, (list, tuple)) and len(model) == 2:
        maybe_cols = model[1]
        if isinstance(maybe_cols, (list, tuple)) and all(isinstance(c, str) for c in maybe_cols):
            return model[0], list(maybe_cols)
    return model, _maybe_extract_features(model)


def _iter_ensemble(model: Any) -> List[Any] | None:
    if isinstance(model, dict):
        if "models" in model and isinstance(model["models"], (list, tuple)):
            return list(model["models"])
        if model and all(hasattr(v, "predict") for v in model.values()):
            return list(model.values())
    if isinstance(model, (list, tuple)) and model and all(hasattr(v, "predict") for v in model):
        return list(model)
    return None


def _rank_uniform(values: pd.Series) -> pd.Series:
    ranked = values.rank(pct=True, method="first")
    return ranked.clip(0.0, 1.0)


def _default_feature_cols(live_universe: pd.DataFrame) -> List[str]:
    feature_cols = [c for c in live_universe.columns if c.startswith("feature")]
    if feature_cols:
        return feature_cols
    exclude = {"symbol", "date", "era", "target", "prediction"}
    numeric_cols = [
        c for c in live_universe.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(live_universe[c])
    ]
    if numeric_cols:
        return numeric_cols
    return [c for c in live_universe.columns if c not in exclude]


def _predict_with_model(model: Any, live_universe: pd.DataFrame) -> pd.Series:
    ensemble = _iter_ensemble(model)
    if ensemble:
        feature_cols = _maybe_extract_features(ensemble[0]) or _default_feature_cols(live_universe)
        df = live_universe.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        features = df[feature_cols].astype(np.float32, copy=False)
        preds = []
        for mdl in ensemble:
            preds.append(pd.Series(mdl.predict(features), index=features.index))
        return pd.concat(preds, axis=1).mean(axis=1)

    if hasattr(model, "predict"):
        feature_cols = _maybe_extract_features(model) or _default_feature_cols(live_universe)
        df = live_universe.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        features = df[feature_cols].astype(np.float32, copy=False)
        preds = model.predict(features)
        return pd.Series(preds, index=features.index)

    if callable(model):
        output = model(live_universe)
        if isinstance(output, pd.DataFrame):
            if "prediction" in output.columns:
                return output["prediction"].copy()
            if output.shape[1] == 1:
                return output.iloc[:, 0].copy()
        if isinstance(output, pd.Series):
            return output.copy()
        return pd.Series(output, index=live_universe.index)

    raise TypeError("Unsupported model type for prediction")


def main() -> None:
    logging.info("Downloading live crypto universe")
    candidates: List[str] = []
    if LIVE_DATASET:
        candidates.append(LIVE_DATASET)
    if LIVE_FILENAME:
        if "/" in LIVE_FILENAME:
            candidates.append(LIVE_FILENAME)
        else:
            candidates.append(f"{DATA_VERSION}/{LIVE_FILENAME}")
    else:
        candidates.extend(
            [
                f"{DATA_VERSION}/live.parquet",
                f"{DATA_VERSION}/live_universe.parquet",
                f"v2.0/live.parquet",
                f"v2.0/live_universe.parquet",
                f"crypto/v2.0/live.parquet",
                f"crypto/v2.0/live_universe.parquet",
            ]
        )
    dataset_used = _download_first_available(candidates)
    live_universe = pd.read_parquet(dataset_used)

    model = _safe_load_model(MODEL_PATH)
    model, explicit_cols = _unwrap_model_and_features(model)
    if explicit_cols:
        for col in explicit_cols:
            if col not in live_universe.columns:
                live_universe[col] = 0.0

    preds = _predict_with_model(model, live_universe)
    if RANK_UNIFORM:
        preds = _rank_uniform(preds)

    if "symbol" in live_universe.columns:
        index = live_universe["symbol"]
        index.name = "symbol"
    else:
        index = live_universe.index
    predictions = pd.DataFrame({"prediction": preds.values}, index=index)

    logging.info("Writing predictions and submitting")
    predictions.to_csv(PREDICTIONS_PATH, index=True)
    napi.upload_predictions(PREDICTIONS_PATH, model_id=MODEL_ID, tournament=TOURNAMENT)


if __name__ == "__main__":
    main()
