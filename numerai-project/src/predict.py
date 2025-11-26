"""Prediction entrypoint for Numerai models with stacking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import model_lgb, model_mlp, model_ridge, stacker, utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction."""
    parser = argparse.ArgumentParser(description="Generate Numerai predictions")
    parser.add_argument("--params", default="config/model_params.yaml", help="Path to model params")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training config")
    parser.add_argument("--features", default="config/features.yaml", help="Path to feature config")
    return parser.parse_args()


def load_configs(config_path: str, params_path: str, features_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load configuration files."""
    training_cfg = utils.load_yaml(config_path)
    params_cfg = utils.normalize_params(utils.load_yaml(params_path))
    features_cfg = utils.load_yaml(features_path)
    return training_cfg or {}, params_cfg or {}, features_cfg or {}


def model_factories(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return builders for base models."""
    return {
        "lgb": lambda: model_lgb.LightGBMModel(params.get("lightgbm", {})),
        "ridge": lambda: model_ridge.RidgeModel(params.get("ridge", {})),
        "mlp": lambda: model_mlp.MLPModel(params.get("mlp", {})),
    }


def load_models(models_dir: Path, params: Dict[str, Any]) -> Tuple[Dict[str, Any], stacker.ModelStacker]:
    """Load fitted models from disk or build fresh placeholders."""
    models: Dict[str, Any] = {}
    for name, builder in model_factories(params).items():
        path = models_dir / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
            utils.log(f"Loaded model: {path}")
        else:
            models[name] = builder()
            utils.log(f"Model not found, using fresh {name} instance")

    stacker_path = models_dir / "stacker.pkl"
    if stacker_path.exists():
        stk = joblib.load(stacker_path)
        utils.log(f"Loaded stacker: {stacker_path}")
    else:
        stk = stacker.ModelStacker(params.get("stacker", {"alpha": 0.5}))
        utils.log("Stacker not found, using fresh instance")
    return models, stk


def load_features(training_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, Path]:
    """Load tournament features and ids."""
    tournament_path = Path(training_cfg.get("files", {}).get("tournament", "data/numerai_tournament_data.parquet"))
    if not tournament_path.is_absolute():
        tournament_path = PROJECT_ROOT / tournament_path
    feature_cfg = features_cfg.get("features", {})
    feature_prefix = feature_cfg.get("prefix", "feature")
    feature_limit = feature_cfg.get("max_features") or feature_cfg.get("limit")
    schema_cols = utils.parquet_columns(tournament_path)
    feature_cols = [c for c in schema_cols if c.startswith(feature_prefix)] if schema_cols else []
    if feature_limit:
        feature_cols = feature_cols[:feature_limit]
    id_candidates = ["id", "prediction_id", "row_id", "tournament_id"]
    id_col = next((c for c in id_candidates if c in schema_cols), None) if schema_cols else None
    columns_to_read = feature_cols + ([id_col] if id_col else [])
    df = utils.safe_read_parquet(tournament_path, columns=columns_to_read or None)
    if id_col and id_col not in df.columns and df.index.name == id_col:
        df = df.reset_index()
    if not feature_cols:
        feature_cols = utils.get_feature_columns(df, feature_prefix)
        if feature_limit:
            feature_cols = feature_cols[:feature_limit]

    if df.empty or not feature_cols:
        utils.log("Tournament data missing or feature columns not found; using dummy data.")
        df, _ = utils.dummy_dataset(prefix=feature_prefix)
        feature_cols = utils.get_feature_columns(df, feature_prefix)

    if id_col is None:
        for candidate in id_candidates:
            if candidate in df.columns:
                id_col = candidate
                break
    if id_col is None:
        # fallback: utilise la première colonne non-feature ou un index
        non_feature_cols = [c for c in df.columns if c not in feature_cols]
        ids = df[non_feature_cols[0]].reset_index(drop=True) if non_feature_cols else pd.Series(range(len(df)), name="id")
    else:
        ids = df[id_col].reset_index(drop=True)

    features_df = df[feature_cols].astype(np.float32, copy=False)
    del df
    return features_df, ids, Path(tournament_path)


def predict(models_dir: Path, training_cfg: Dict[str, Any], params_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Generate predictions and save submission file."""
    features_df, ids, tournament_path = load_features(training_cfg, features_cfg)
    models, stk = load_models(models_dir, params_cfg)

    preds: Dict[str, pd.Series] = {}
    for name, mdl in models.items():
        preds[name] = mdl.predict(features_df).reset_index(drop=True)
        utils.log(f"Generated predictions for {name}")

    base_pred_df = pd.DataFrame(preds)
    try:
        final_pred = stk.predict(base_pred_df)
    except Exception:
        final_pred = base_pred_df.mean(axis=1)
    submission = pd.DataFrame({"id": ids, "prediction": final_pred.values})

    submission_path = Path(training_cfg.get("files", {}).get("submission", "submission.csv"))
    if not submission_path.is_absolute():
        submission_path = PROJECT_ROOT / submission_path
    utils.save_submission(submission, submission_path)
    utils.log(f"Predictions ready from data: {tournament_path}")
    return submission


def main() -> None:
    args = parse_args()
    training_cfg, params_cfg, features_cfg = load_configs(args.config, args.params, args.features)
    models_dir = PROJECT_ROOT / "models"
    predict(models_dir, training_cfg, params_cfg, features_cfg)


if __name__ == "__main__":
    main()
