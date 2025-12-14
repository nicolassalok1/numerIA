"""Prediction entrypoint for Numerai models with stacking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, List

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
    parser.add_argument("--params", default="config/program_input_params.yaml", help="Path to model params")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training config")
    parser.add_argument("--features", default="config/features.yaml", help="Path to feature config")
    return parser.parse_args()


def load_configs(config_path: str, params_path: str, features_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load configuration files."""
    training_cfg = utils.load_yaml(config_path)
    params_cfg = utils.normalize_params(utils.load_yaml(params_path))
    features_cfg = utils.load_yaml(features_path)
    return training_cfg or {}, params_cfg or {}, features_cfg or {}


def load_model_spec(models_dir: Path) -> Dict[str, Any]:
    """Load model spec emitted during training (base models + aggregator)."""
    path = models_dir / "model_spec.json"
    if not path.exists():
        return {}
    try:
        return utils.load_json(path) or {}
    except Exception:
        return {}


def builder_for_model(name: str, params: Dict[str, Any]) -> Any:
    """Create a fresh model instance matching a persisted name."""
    if name == "ridge":
        return model_ridge.RidgeModel(params.get("ridge", {}))
    if name == "mlp":
        return model_mlp.MLPModel(params.get("mlp", {}))
    if name.startswith("lgb"):
        lgb_params = dict(params.get("lightgbm", {}) or {})
        if name.startswith("lgb_s"):
            try:
                seed = int(name.split("lgb_s", 1)[1])
                lgb_params["seed"] = seed
                lgb_params["data_random_seed"] = seed
                lgb_params["feature_fraction_seed"] = seed
                lgb_params["bagging_seed"] = seed
            except Exception:
                pass
        return model_lgb.LightGBMModel(lgb_params)
    return model_lgb.LightGBMModel(params.get("lightgbm", {}))


def load_models(models_dir: Path, params: Dict[str, Any]) -> Tuple[Dict[str, Any], stacker.ModelStacker | None, str]:
    """Load fitted base models and optional stacker."""
    spec = load_model_spec(models_dir)
    base_models = spec.get("base_models")
    aggregator = str(spec.get("aggregator", "ridge")).strip().lower()
    if aggregator not in {"mean", "ridge"}:
        aggregator = "ridge"

    if not base_models:
        base_models = ["lgb", "ridge", "mlp"]

    models: Dict[str, Any] = {}
    for name in base_models:
        path = models_dir / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
            utils.log(f"Loaded model: {path}")
        else:
            models[name] = builder_for_model(name, params)
            utils.log(f"Model not found, using fresh {name} instance")

    stk: stacker.ModelStacker | None = None
    if aggregator == "ridge":
        stacker_path = models_dir / "stacker.pkl"
        if stacker_path.exists():
            stk = joblib.load(stacker_path)
            utils.log(f"Loaded stacker: {stacker_path}")
        else:
            utils.log("Stacker not found; falling back to mean aggregation.")
            aggregator = "mean"

    return models, stk, aggregator


def load_feature_columns(models_dir: Path, training_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> Tuple[List[str], str]:
    """Resolve feature columns (prefer persisted list from training)."""
    persisted = models_dir / "feature_columns.json"
    if persisted.exists():
        data = utils.load_json(persisted)
        cols = data.get("feature_cols") or []
        prefix = data.get("feature_prefix") or features_cfg.get("features", {}).get("prefix", "feature")
        if cols:
            utils.log(f"Using persisted feature columns: {persisted}")
            return list(cols), str(prefix)

    feature_cfg = features_cfg.get("features", {})
    prefix = str(feature_cfg.get("prefix", "feature"))
    return [], prefix


def load_features(models_dir: Path, training_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.Series | None, Path]:
    """Load tournament features (+ ids + eras if available)."""
    tournament_path = Path(training_cfg.get("files", {}).get("tournament", "data/numerai_tournament_data.parquet"))
    if not tournament_path.is_absolute():
        tournament_path = PROJECT_ROOT / tournament_path
    schema_cols = utils.parquet_columns(tournament_path)
    persisted_cols, feature_prefix = load_feature_columns(models_dir, training_cfg, features_cfg)

    if persisted_cols:
        requested_feature_cols = persisted_cols
    else:
        feature_cfg = features_cfg.get("features", {})
        feature_limit = feature_cfg.get("max_features") or feature_cfg.get("limit")
        requested_feature_cols = [c for c in schema_cols if c.startswith(feature_prefix)] if schema_cols else []
        if feature_limit:
            requested_feature_cols = requested_feature_cols[: int(feature_limit)]

    available_feature_cols = [c for c in requested_feature_cols if c in (schema_cols or [])]
    id_candidates = ["id", "prediction_id", "row_id", "tournament_id"]
    id_col = next((c for c in id_candidates if c in schema_cols), None) if schema_cols else None
    columns_to_read = available_feature_cols + ([id_col] if id_col else [])
    if schema_cols and "era" in schema_cols:
        columns_to_read.append("era")
    df = utils.safe_read_parquet(tournament_path, columns=columns_to_read or None)
    if id_col and id_col not in df.columns and df.index.name == id_col:
        df = df.reset_index()

    if df.empty or not requested_feature_cols:
        utils.log("Tournament data missing or feature columns not found; using dummy data.")
        df, _ = utils.dummy_dataset(prefix=feature_prefix)
        requested_feature_cols = utils.get_feature_columns(df, feature_prefix)

    if id_col is None:
        for candidate in id_candidates:
            if candidate in df.columns:
                id_col = candidate
                break
    if id_col is None:
        # fallback: utilise la première colonne non-feature ou un index
        non_feature_cols = [c for c in df.columns if c not in requested_feature_cols]
        ids = df[non_feature_cols[0]].reset_index(drop=True) if non_feature_cols else pd.Series(range(len(df)), name="id")
    else:
        ids = df[id_col].reset_index(drop=True)

    # Add missing features as zeros, then enforce stable column order
    missing = [c for c in requested_feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    features_df = df[list(requested_feature_cols)].astype(np.float32, copy=False)
    eras = df["era"].reset_index(drop=True) if "era" in df.columns else None
    del df
    return features_df, ids, eras, Path(tournament_path)


def predict(models_dir: Path, training_cfg: Dict[str, Any], params_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Generate predictions and save submission file."""
    features_df, ids, eras, tournament_path = load_features(models_dir, training_cfg, features_cfg)
    models, stk, aggregator = load_models(models_dir, params_cfg)

    preds: Dict[str, pd.Series] = {}
    for name, mdl in models.items():
        preds[name] = mdl.predict(features_df).reset_index(drop=True)
        utils.log(f"Generated predictions for {name}")

    base_pred_df = pd.DataFrame(preds)
    if aggregator == "ridge" and stk is not None:
        try:
            final_pred = stk.predict(base_pred_df)
        except Exception:
            final_pred = base_pred_df.mean(axis=1)
    else:
        final_pred = base_pred_df.mean(axis=1)
    submission = pd.DataFrame({"id": ids, "prediction": final_pred.values})

    # Post-processing: ranking improves stability and often boosts corr
    pred_cfg = (training_cfg.get("prediction", {}) or {})
    do_rank = pred_cfg.get("rank", True)
    rank_by_era = pred_cfg.get("rank_by_era", True)
    if do_rank:
        if rank_by_era and eras is not None:
            submission["prediction"] = utils.rank_uniform(submission["prediction"], era=eras)
        else:
            submission["prediction"] = utils.rank_uniform(submission["prediction"])

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
