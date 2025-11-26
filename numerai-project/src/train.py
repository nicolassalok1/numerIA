"""Training entrypoint for Numerai models with KFold stacking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import model_lgb, model_mlp, model_ridge, stacker, utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train Numerai models with stacking")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training config")
    parser.add_argument("--params", default="config/model_params.yaml", help="Path to model params")
    parser.add_argument("--features", default="config/features.yaml", help="Path to feature config")
    return parser.parse_args()


def load_configs(config_path: str, params_path: str, features_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load training, model param, and feature configs."""
    training_cfg = utils.load_yaml(config_path)
    params_cfg = utils.load_yaml(params_path)
    features_cfg = utils.load_yaml(features_path)
    return training_cfg or {}, params_cfg or {}, features_cfg or {}


def prepare_data(training_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load training data and return features, target, and feature prefix."""
    feature_cfg = features_cfg.get("features", {})
    feature_prefix = feature_cfg.get("prefix", "feature")
    feature_limit = feature_cfg.get("max_features") or feature_cfg.get("limit")
    row_limit = training_cfg.get("general", {}).get("row_limit")
    seed = training_cfg.get("general", {}).get("seed", 42)
    train_path = Path(training_cfg.get("files", {}).get("train", "data/numerai_training_data.parquet"))
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    schema_cols = utils.parquet_columns(train_path)
    feature_cols = [c for c in schema_cols if c.startswith(feature_prefix)] if schema_cols else []
    if feature_limit:
        feature_cols = feature_cols[:feature_limit]
    target_col = utils.find_target_column(schema_cols) if schema_cols else None
    columns_to_read = feature_cols + ([target_col] if target_col else [])
    df = utils.safe_read_parquet(train_path, columns=columns_to_read or None)

    if df.empty:
        utils.log("Training data missing; using dummy dataset.")
        X, y = utils.dummy_dataset(prefix=feature_prefix)
        return X, y, feature_prefix

    if not feature_cols:
        feature_cols = utils.get_feature_columns(df, feature_prefix)
        if feature_limit:
            feature_cols = feature_cols[:feature_limit]
    if not feature_cols:
        utils.log("No feature columns found; using dummy dataset.")
        X, y = utils.dummy_dataset(prefix=feature_prefix)
        return X, y, feature_prefix
    if row_limit and len(df) > row_limit:
        df = df.sample(n=row_limit, random_state=seed).reset_index(drop=True)
        utils.log(f"Row limit applied: {row_limit} rows sampled for training.")

    if not target_col:
        target_col = utils.detect_target(df)
    if target_col not in df.columns:
        utils.log("Target column missing; using zero target.")
        y = pd.Series(np.zeros(len(df), dtype=np.float32), name="target")
    else:
        y = df[target_col].astype(np.float32, copy=False)

    X = df[feature_cols].astype(np.float32, copy=False)
    del df
    return X, y, feature_prefix


def model_factories(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return builders for base models."""
    return {
        "lgb": lambda: model_lgb.LightGBMModel(params.get("lightgbm", {})),
        "ridge": lambda: model_ridge.RidgeModel(params.get("ridge", {})),
        "mlp": lambda: model_mlp.MLPModel(params.get("mlp", {})),
    }


def train_base_models(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], n_folds: int, seed: int, models_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train base models with KFold and return fitted models and OOF predictions."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = pd.DataFrame(index=range(len(X)))
    fitted_models: Dict[str, Any] = {}

    for name, builder in model_factories(params).items():
        utils.log(f"Training {name} with {n_folds} folds")
        fold_pred = np.zeros(len(X))
        for train_idx, val_idx in kf.split(X):
            mdl = builder()
            mdl.train(X.iloc[train_idx], y.iloc[train_idx])
            fold_pred[val_idx] = mdl.predict(X.iloc[val_idx])
        oof_preds[name] = fold_pred

        final_model = builder()
        final_model.train(X, y)
        fitted_models[name] = final_model
        save_model(models_dir / f"{name}.pkl", final_model)

    return fitted_models, oof_preds


def train_stacker(oof_preds: pd.DataFrame, y: pd.Series, params: Dict[str, Any], models_dir: Path) -> stacker.ModelStacker:
    """Train ridge stacker on OOF predictions."""
    stack_params = params.get("stacker", {"alpha": 0.5})
    stk = stacker.ModelStacker(stack_params)
    stk.fit(oof_preds, y)
    save_model(models_dir / "stacker.pkl", stk)
    return stk


def save_model(path: Path, model: Any) -> None:
    """Persist model to disk."""
    utils.ensure_dir(path.parent)
    joblib.dump(model, path)
    utils.log(f"Saved model: {path}")


def main() -> None:
    """Main training routine with stacking."""
    args = parse_args()
    training_cfg, params_cfg, features_cfg = load_configs(args.config, args.params, args.features)

    X, y, _ = prepare_data(training_cfg, features_cfg)
    n_folds = training_cfg.get("general", {}).get("n_folds", 5)
    seed = training_cfg.get("general", {}).get("seed", 42)
    models_dir = PROJECT_ROOT / "models"

    _, oof_preds = train_base_models(X, y, params_cfg, n_folds, seed, models_dir)
    _ = train_stacker(oof_preds, y, params_cfg, models_dir)
    utils.log("Training complete.")


if __name__ == "__main__":
    main()
